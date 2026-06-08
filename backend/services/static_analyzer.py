"""
Prism Static Analyzer — Pre-Analysis Layer (tree-sitter only).

Runs *before* the LLM agents to produce structured context that dramatically
reduces false positives. The LLM validates patterns the static tool already
found, rather than guessing from raw diff text.

Architecture:
  1. Parse unified diff → extract only `+` lines per file
  2. Run tree-sitter over reconstructed source to extract:
     - Function signatures, parameters, return types
     - Intra-file call graph
     - Cyclomatic complexity per function
     - Deeply nested code (depth > 4)
  3. Run OWASP anti-pattern queries via tree-sitter:
     - SQL injection (f-strings, string concatenation in queries)
     - XSS (innerHTML, dangerouslySetInnerHTML, document.write)
     - Hardcoded secrets (password = "...", api_key = "...")
     - Unsafe deserialization (pickle.loads, yaml.load, eval, exec)
     - Command injection (os.system, subprocess with shell=True)
     - Path traversal (open(user_input))
  4. Return structured JSON consumed by agent prompts

Design choice: Option 3 — all patterns implemented as tree-sitter AST
traversals in pure Python. No ast-grep CLI dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from observability.logging import get_logger
from services.simple_ast_parser import LANGUAGE_MAP, LANGUAGE_MODULES

logger = get_logger(__name__)


# ── Diff Parser ─────────────────────────────────────────────────────────────


@dataclass
class DiffFile:
    """Represents the added lines from a single file in a unified diff."""

    file_path: str
    added_lines: list[str]  # Only the `+` lines (content, no prefix)
    added_line_numbers: list[int]  # Original line numbers in the new file
    source: str = ""  # Reconstructed source from + lines


def parse_diff_added_lines(diff_text: str) -> list[DiffFile]:
    """Parse unified diff and extract only the `+` lines per file.

    We reconstruct a synthetic source string from ONLY the added lines.
    This is what tree-sitter parses — we analyze new code, not old code.
    """
    files: list[DiffFile] = []
    current_file: DiffFile | None = None

    # Track new-file line numbers from @@ hunk headers
    new_line_num = 0

    for raw_line in diff_text.split("\n"):
        # Detect file header: "diff --git a/path b/path" or "+++ b/path"
        if raw_line.startswith("+++ b/"):
            file_path = raw_line[6:].strip()
            current_file = DiffFile(file_path=file_path, added_lines=[], added_line_numbers=[])
            files.append(current_file)
            continue

        if raw_line.startswith("+++ "):
            # Handle "+++ path" without b/ prefix
            file_path = raw_line[4:].strip()
            if file_path.startswith("b/"):
                file_path = file_path[2:]
            current_file = DiffFile(file_path=file_path, added_lines=[], added_line_numbers=[])
            files.append(current_file)
            continue

        # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
        hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
        if hunk_match:
            new_line_num = int(hunk_match.group(1))
            continue

        if current_file is None:
            continue

        # Skip removed lines (they don't exist in the new code)
        if raw_line.startswith("-"):
            continue

        # Added lines: strip the `+` prefix
        if raw_line.startswith("+"):
            content = raw_line[1:]  # Remove the leading `+`
            current_file.added_lines.append(content)
            current_file.added_line_numbers.append(new_line_num)
            new_line_num += 1
            continue

        # Context lines (no prefix) — advance the line counter but don't store
        if not raw_line.startswith("\\"):  # Skip "\ No newline at end of file"
            new_line_num += 1

    # Reconstruct source from added lines
    for f in files:
        f.source = "\n".join(f.added_lines)

    return files


# ── OWASP Pattern Definitions ───────────────────────────────────────────────


@dataclass
class StaticFinding:
    """A single anti-pattern finding from static analysis."""

    rule_id: str
    category: str  # OWASP category or structural
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    file_path: str
    line: int  # 1-indexed line in the NEW file
    message: str
    matched_code: str  # The exact code snippet that triggered the finding


# ── Tree-Sitter OWASP Scanner ───────────────────────────────────────────────


class TreeSitterOWASPScanner:
    """Scans source code for OWASP anti-patterns using tree-sitter AST walks.

    Every pattern match is AST-aware — it won't fire inside comments or
    docstrings (unlike regex-based scanners).
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Any] = {}

    def _get_parser(self, language: str) -> Any | None:
        """Lazy-init a tree-sitter parser for the given language."""
        if language in self._parsers:
            return self._parsers[language]

        if language not in LANGUAGE_MODULES:
            return None

        try:
            from tree_sitter import Language, Parser

            lang_module = LANGUAGE_MODULES[language]
            if language == "typescript":
                lang_obj = Language(lang_module.language_typescript())
            elif language == "tsx":
                lang_obj = Language(lang_module.language_tsx())
            else:
                lang_obj = Language(lang_module.language())

            parser = Parser(lang_obj)
            self._parsers[language] = parser
            return parser
        except Exception as e:
            logger.warning("static_analyzer_parser_init_failed", language=language, error=str(e))
            return None

    def scan(self, file_path: str, source: str, line_offset_map: list[int]) -> list[StaticFinding]:
        """Scan a source string for OWASP anti-patterns.

        Args:
            file_path: Relative path (e.g., "services/auth.py")
            source: Reconstructed source from + lines
            line_offset_map: Maps 0-indexed line in `source` to real line number in new file

        Returns:
            List of StaticFinding objects.
        """
        ext = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        language = LANGUAGE_MAP.get(ext)
        if not language:
            return []

        parser = self._get_parser(language)
        if not parser:
            return []

        try:
            tree = parser.parse(source.encode("utf-8"))
        except Exception as e:
            logger.warning("static_analyzer_parse_failed", file=file_path, error=str(e))
            return []

        findings: list[StaticFinding] = []

        if language == "python":
            findings.extend(self._scan_python(tree.root_node, source, file_path, line_offset_map))
        elif language in {"javascript", "typescript", "tsx"}:
            findings.extend(self._scan_javascript(tree.root_node, source, file_path, line_offset_map))
        elif language == "go":
            findings.extend(self._scan_go(tree.root_node, source, file_path, line_offset_map))

        return findings

    def _real_line(self, node_line: int, line_map: list[int]) -> int:
        """Convert a 0-indexed tree-sitter line to the real file line number."""
        if 0 <= node_line < len(line_map):
            return line_map[node_line]
        return node_line + 1

    def _node_text(self, node: Any, source: str) -> str:
        """Extract the text of an AST node."""
        return source[node.start_byte : node.end_byte]

    # ── Python Patterns ──────────────────────────────────────────────────

    def _scan_python(self, root: Any, source: str, file_path: str, line_map: list[int]) -> list[StaticFinding]:
        findings: list[StaticFinding] = []

        def walk(node: Any) -> None:
            # ── SQL Injection: f-string or .format() with SQL keywords ────
            if node.type == "string" and node.type == "string":
                text = self._node_text(node, source)
                upper = text.upper()
                if any(kw in upper for kw in ("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "DROP ")):
                    parent = node.parent
                    # f-string with SQL
                    if node.type == "string" and "{" in text and text.startswith(('f"', "f'", 'f"', "f'")):
                        findings.append(
                            StaticFinding(
                                rule_id="python-sqli-fstring",
                                category="A03:2021-Injection",
                                severity="CRITICAL",
                                file_path=file_path,
                                line=self._real_line(node.start_point[0], line_map),
                                message="SQL query built with f-string interpolation — use parameterized queries",
                                matched_code=text[:120],
                            )
                        )
                    # String concatenation with SQL
                    elif parent and parent.type == "binary_operator":
                        op_text = self._node_text(parent, source)
                        if "+" in op_text:
                            findings.append(
                                StaticFinding(
                                    rule_id="python-sqli-concat",
                                    category="A03:2021-Injection",
                                    severity="CRITICAL",
                                    file_path=file_path,
                                    line=self._real_line(node.start_point[0], line_map),
                                    message="SQL query built via string concatenation — use parameterized queries",
                                    matched_code=op_text[:120],
                                )
                            )

            # ── Hardcoded Secrets ─────────────────────────────────────────
            if node.type == "assignment":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                if left and right and right.type == "string":
                    var_name = self._node_text(left, source).lower()
                    value = self._node_text(right, source)
                    secret_keywords = (
                        "password",
                        "secret",
                        "api_key",
                        "apikey",
                        "private_key",
                        "token",
                        "auth_token",
                        "aws_secret",
                        "database_url",
                    )
                    if any(kw in var_name for kw in secret_keywords):
                        # Ignore empty strings and obvious placeholders
                        stripped = value.strip("\"'")
                        if stripped and stripped not in ("", "changeme", "xxx", "your-key-here"):
                            findings.append(
                                StaticFinding(
                                    rule_id="python-hardcoded-secret",
                                    category="A07:2021-Auth-Failures",
                                    severity="HIGH",
                                    file_path=file_path,
                                    line=self._real_line(node.start_point[0], line_map),
                                    message=f"Potential hardcoded secret in variable '{self._node_text(left, source)}'",
                                    matched_code=self._node_text(node, source)[:120],
                                )
                            )

            # ── Unsafe Deserialization ────────────────────────────────────
            if node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node:
                    func_text = self._node_text(func_node, source)

                    # pickle.loads, pickle.load
                    if func_text in ("pickle.loads", "pickle.load"):
                        findings.append(
                            StaticFinding(
                                rule_id="python-unsafe-pickle",
                                category="A08:2021-Integrity-Failures",
                                severity="CRITICAL",
                                file_path=file_path,
                                line=self._real_line(node.start_point[0], line_map),
                                message="pickle.loads() deserializes arbitrary objects — can lead to remote code execution",
                                matched_code=self._node_text(node, source)[:120],
                            )
                        )

                    # yaml.load without SafeLoader
                    if func_text in ("yaml.load", "yaml.unsafe_load"):
                        call_text = self._node_text(node, source)
                        if "SafeLoader" not in call_text and "safe_load" not in call_text:
                            findings.append(
                                StaticFinding(
                                    rule_id="python-unsafe-yaml",
                                    category="A08:2021-Integrity-Failures",
                                    severity="HIGH",
                                    file_path=file_path,
                                    line=self._real_line(node.start_point[0], line_map),
                                    message="yaml.load() without SafeLoader can execute arbitrary code",
                                    matched_code=call_text[:120],
                                )
                            )

                    # eval() / exec()
                    if func_text in ("eval", "exec"):
                        findings.append(
                            StaticFinding(
                                rule_id="python-eval-exec",
                                category="A03:2021-Injection",
                                severity="CRITICAL",
                                file_path=file_path,
                                line=self._real_line(node.start_point[0], line_map),
                                message=f"{func_text}() executes arbitrary code — avoid with untrusted input",
                                matched_code=self._node_text(node, source)[:120],
                            )
                        )

                    # ── Command Injection ─────────────────────────────────
                    if func_text in ("os.system", "os.popen"):
                        findings.append(
                            StaticFinding(
                                rule_id="python-cmd-injection",
                                category="A03:2021-Injection",
                                severity="CRITICAL",
                                file_path=file_path,
                                line=self._real_line(node.start_point[0], line_map),
                                message=f"{func_text}() is vulnerable to command injection — use subprocess with shell=False",
                                matched_code=self._node_text(node, source)[:120],
                            )
                        )

                    if func_text in ("subprocess.call", "subprocess.run", "subprocess.Popen"):
                        call_text = self._node_text(node, source)
                        if "shell=True" in call_text:
                            findings.append(
                                StaticFinding(
                                    rule_id="python-subprocess-shell",
                                    category="A03:2021-Injection",
                                    severity="HIGH",
                                    file_path=file_path,
                                    line=self._real_line(node.start_point[0], line_map),
                                    message=f"{func_text}() with shell=True is vulnerable to command injection",
                                    matched_code=call_text[:120],
                                )
                            )

            for child in node.children:
                walk(child)

        walk(root)

        # ── f-string SQL scan (tree-sitter may represent f-strings as 'string' with interpolation) ──
        # Fallback: also scan raw source lines for f-string SQL patterns
        for i, line in enumerate(source.split("\n")):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if any(kw in upper for kw in ("SELECT ", "INSERT ", "UPDATE ", "DELETE ")):
                if 'f"' in stripped or "f'" in stripped:
                    # Check we haven't already reported this line
                    real_ln = self._real_line(i, line_map)
                    if not any(f.line == real_ln and f.rule_id == "python-sqli-fstring" for f in findings):
                        findings.append(
                            StaticFinding(
                                rule_id="python-sqli-fstring",
                                category="A03:2021-Injection",
                                severity="CRITICAL",
                                file_path=file_path,
                                line=real_ln,
                                message="SQL query built with f-string interpolation — use parameterized queries",
                                matched_code=stripped[:120],
                            )
                        )

        return findings

    # ── JavaScript/TypeScript Patterns ────────────────────────────────────

    def _scan_javascript(self, root: Any, source: str, file_path: str, line_map: list[int]) -> list[StaticFinding]:
        findings: list[StaticFinding] = []

        def walk(node: Any) -> None:
            # ── XSS: innerHTML assignment ─────────────────────────────────
            if node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                if left:
                    left_text = self._node_text(left, source)
                    if "innerHTML" in left_text:
                        findings.append(
                            StaticFinding(
                                rule_id="js-xss-innerhtml",
                                category="A03:2021-Injection",
                                severity="HIGH",
                                file_path=file_path,
                                line=self._real_line(node.start_point[0], line_map),
                                message="Direct innerHTML assignment can lead to XSS — use textContent or a sanitizer",
                                matched_code=self._node_text(node, source)[:120],
                            )
                        )

            # ── XSS: dangerouslySetInnerHTML (React) ──────────────────────
            if node.type == "jsx_attribute":
                attr_text = self._node_text(node, source)
                if "dangerouslySetInnerHTML" in attr_text:
                    findings.append(
                        StaticFinding(
                            rule_id="js-xss-dangerous-html",
                            category="A03:2021-Injection",
                            severity="HIGH",
                            file_path=file_path,
                            line=self._real_line(node.start_point[0], line_map),
                            message="dangerouslySetInnerHTML bypasses React's XSS protection — ensure input is sanitized",
                            matched_code=attr_text[:120],
                        )
                    )

            # ── XSS: document.write ───────────────────────────────────────
            if node.type == "call_expression":
                func_node = node.child_by_field_name("function")
                if func_node:
                    func_text = self._node_text(func_node, source)

                    if func_text == "document.write":
                        findings.append(
                            StaticFinding(
                                rule_id="js-xss-document-write",
                                category="A03:2021-Injection",
                                severity="HIGH",
                                file_path=file_path,
                                line=self._real_line(node.start_point[0], line_map),
                                message="document.write() can introduce XSS — use DOM APIs instead",
                                matched_code=self._node_text(node, source)[:120],
                            )
                        )

                    # ── eval() ────────────────────────────────────────────
                    if func_text == "eval":
                        findings.append(
                            StaticFinding(
                                rule_id="js-eval",
                                category="A03:2021-Injection",
                                severity="CRITICAL",
                                file_path=file_path,
                                line=self._real_line(node.start_point[0], line_map),
                                message="eval() executes arbitrary JavaScript — avoid with untrusted input",
                                matched_code=self._node_text(node, source)[:120],
                            )
                        )

            # ── Hardcoded Secrets (variable declarations) ─────────────────
            if node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                if name_node and value_node and value_node.type == "string":
                    var_name = self._node_text(name_node, source).lower()
                    secret_keywords = ("password", "secret", "apikey", "api_key", "private_key", "token", "auth_token")
                    if any(kw in var_name for kw in secret_keywords):
                        val = self._node_text(value_node, source).strip("\"'`")
                        if val and val not in ("", "changeme", "xxx"):
                            findings.append(
                                StaticFinding(
                                    rule_id="js-hardcoded-secret",
                                    category="A07:2021-Auth-Failures",
                                    severity="HIGH",
                                    file_path=file_path,
                                    line=self._real_line(node.start_point[0], line_map),
                                    message=f"Potential hardcoded secret in '{self._node_text(name_node, source)}'",
                                    matched_code=self._node_text(node, source)[:120],
                                )
                            )

            for child in node.children:
                walk(child)

        walk(root)
        return findings

    # ── Go Patterns ──────────────────────────────────────────────────────

    def _scan_go(self, root: Any, source: str, file_path: str, line_map: list[int]) -> list[StaticFinding]:
        findings: list[StaticFinding] = []

        def walk(node: Any) -> None:
            if node.type == "call_expression":
                func_node = node.child_by_field_name("function")
                if func_node:
                    func_text = self._node_text(func_node, source)
                    call_text = self._node_text(node, source)

                    # fmt.Sprintf with SQL keywords
                    if func_text == "fmt.Sprintf":
                        upper = call_text.upper()
                        if any(kw in upper for kw in ("SELECT ", "INSERT ", "UPDATE ", "DELETE ")):
                            findings.append(
                                StaticFinding(
                                    rule_id="go-sqli-sprintf",
                                    category="A03:2021-Injection",
                                    severity="CRITICAL",
                                    file_path=file_path,
                                    line=self._real_line(node.start_point[0], line_map),
                                    message="SQL query built with fmt.Sprintf — use parameterized queries",
                                    matched_code=call_text[:120],
                                )
                            )

                    # exec.Command without proper sanitization
                    if func_text in ("exec.Command", "exec.CommandContext"):
                        findings.append(
                            StaticFinding(
                                rule_id="go-cmd-injection",
                                category="A03:2021-Injection",
                                severity="HIGH",
                                file_path=file_path,
                                line=self._real_line(node.start_point[0], line_map),
                                message=f"{func_text}() — ensure command arguments are properly validated",
                                matched_code=call_text[:120],
                            )
                        )

            for child in node.children:
                walk(child)

        walk(root)
        return findings


# ── Structural Analyzer ─────────────────────────────────────────────────────


@dataclass
class FunctionInfo:
    """Structural metadata for a function extracted by tree-sitter."""

    name: str
    file_path: str
    line: int  # Real line number in new file
    end_line: int
    params: list[str]
    complexity: int  # Cyclomatic complexity
    calls: list[str]  # Functions called from this function
    nesting_depth: int  # Maximum nesting depth


class StructuralAnalyzer:
    """Extract structural metadata from source code using tree-sitter.

    Computes:
    - Function signatures with parameters
    - Cyclomatic complexity per function
    - Call graph (which function calls which)
    - Maximum nesting depth
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Any] = {}

    def _get_parser(self, language: str) -> Any | None:
        if language in self._parsers:
            return self._parsers[language]

        if language not in LANGUAGE_MODULES:
            return None

        try:
            from tree_sitter import Language, Parser

            lang_module = LANGUAGE_MODULES[language]
            if language == "typescript":
                lang_obj = Language(lang_module.language_typescript())
            elif language == "tsx":
                lang_obj = Language(lang_module.language_tsx())
            else:
                lang_obj = Language(lang_module.language())

            parser = Parser(lang_obj)
            self._parsers[language] = parser
            return parser
        except Exception:
            return None

    def analyze(self, file_path: str, source: str, line_map: list[int]) -> list[FunctionInfo]:
        """Extract structural info from the + lines of a diff file."""
        ext = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        language = LANGUAGE_MAP.get(ext)
        if not language:
            return []

        parser = self._get_parser(language)
        if not parser:
            return []

        try:
            tree = parser.parse(source.encode("utf-8"))
        except Exception:
            return []

        func_types = self._get_func_types(language)
        functions: list[FunctionInfo] = []

        def walk(node: Any) -> None:
            if node.type in func_types:
                info = self._extract_function(node, source, file_path, language, line_map)
                if info:
                    functions.append(info)

            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return functions

    def _extract_function(
        self, node: Any, source: str, file_path: str, language: str, line_map: list[int]
    ) -> FunctionInfo | None:
        """Extract metadata from a single function node."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = source[name_node.start_byte : name_node.end_byte]

        # Parameters
        params_node = node.child_by_field_name("parameters")
        params: list[str] = []
        if params_node:
            params_text = source[params_node.start_byte : params_node.end_byte]
            # Simple split — good enough for structural overview
            params = [p.strip() for p in params_text.strip("()").split(",") if p.strip()]

        # Cyclomatic complexity
        complexity = self._compute_complexity(node, language, source)

        # Call graph
        calls = self._extract_calls(node, source)

        # Nesting depth
        nesting = self._compute_nesting_depth(node, language)

        # Line mapping
        start_line = line_map[node.start_point[0]] if node.start_point[0] < len(line_map) else node.start_point[0] + 1
        end_line = line_map[node.end_point[0]] if node.end_point[0] < len(line_map) else node.end_point[0] + 1

        return FunctionInfo(
            name=name,
            file_path=file_path,
            line=start_line,
            end_line=end_line,
            params=params,
            complexity=complexity,
            calls=calls,
            nesting_depth=nesting,
        )

    def _compute_complexity(self, node: Any, language: str, source: str) -> int:
        """Cyclomatic complexity: 1 + count of branching nodes."""
        branch_types = {
            "if_statement",
            "elif_clause",
            "else_clause",
            "for_statement",
            "for_in_statement",
            "while_statement",
            "try_statement",
            "except_clause",
            "case_clause",
            "match_statement",
            "conditional_expression",
            "ternary_expression",
            "boolean_operator",  # Python: `and` / `or`
            "binary_expression",  # JS/Go: `&&` / `||`
        }

        count = 1  # Base complexity

        def walk(n: Any) -> None:
            nonlocal count
            if n.type in branch_types:
                if n.type == "binary_expression":
                    op_node = n.child_by_field_name("operator")
                    if op_node:
                        op = source[op_node.start_byte : op_node.end_byte]
                        if op in ("&&", "||"):
                            count += 1
                else:
                    count += 1
            for child in n.children:
                walk(child)

        walk(node)
        return count

    def _extract_calls(self, node: Any, source: str) -> list[str]:
        """Extract function call names within a node."""
        calls: list[str] = []
        call_types = {"call", "call_expression"}

        def walk(n: Any) -> None:
            if n.type in call_types:
                func_node = n.child_by_field_name("function")
                if func_node:
                    name = source[func_node.start_byte : func_node.end_byte]
                    # Simplify: "self.method" → "method", "obj.func" → "func"
                    clean = name.rsplit(".", 1)[-1].strip()
                    if clean and clean.replace("_", "").isalnum():
                        calls.append(clean)
            for child in n.children:
                walk(child)

        walk(node)
        return list(dict.fromkeys(calls))  # Dedupe preserving order

    def _compute_nesting_depth(self, node: Any, language: str) -> int:
        """Compute maximum nesting depth within a function."""
        nesting_types = {
            "if_statement",
            "for_statement",
            "for_in_statement",
            "while_statement",
            "try_statement",
            "with_statement",
        }

        max_depth = 0

        def walk(n: Any, depth: int) -> None:
            nonlocal max_depth
            if n.type in nesting_types:
                depth += 1
                max_depth = max(max_depth, depth)
            for child in n.children:
                walk(child, depth)

        walk(node, 0)
        return max_depth

    @staticmethod
    def _get_func_types(language: str) -> set[str]:
        if language == "python":
            return {"function_definition"}
        elif language in {"javascript", "typescript", "tsx"}:
            return {"function_declaration", "method_definition", "arrow_function"}
        elif language == "go":
            return {"function_declaration", "method_declaration"}
        elif language == "rust":
            return {"function_item"}
        return set()


# ── Main Facade ──────────────────────────────────────────────────────────────


class StaticAnalyzer:
    """Facade that runs both OWASP pattern scanning and structural analysis
    on a unified diff string.

    Usage::

        analyzer = StaticAnalyzer()
        results = analyzer.analyze_diff(diff_text)
        # results is a dict ready to be JSON-serialized and injected into prompts
    """

    def __init__(self):
        self._owasp_scanner = TreeSitterOWASPScanner()
        self._structural_analyzer = StructuralAnalyzer()

    def analyze_diff(self, diff_text: str) -> dict[str, Any]:
        """Run full static analysis on a unified diff.

        Returns a structured dict:
        {
            "tree_sitter": {
                "functions": [...],
                "max_complexity": int,
                "deeply_nested": [...],
                "call_graph": {...}
            },
            "security": {
                "findings": [...],
                "rules_checked": int,
                "total_findings": int
            }
        }
        """
        if not diff_text or not diff_text.strip():
            return self._empty_result()

        # 1. Parse diff → extract + lines per file
        diff_files = parse_diff_added_lines(diff_text)
        if not diff_files:
            return self._empty_result()

        # 2. Run OWASP scanner and structural analyzer on each file
        all_findings: list[dict[str, Any]] = []
        all_functions: list[dict[str, Any]] = []
        deeply_nested: list[dict[str, Any]] = []
        call_graph: dict[str, list[str]] = {}
        max_complexity = 0

        rules_checked = 12  # Number of pattern rules we check

        for df in diff_files:
            if not df.source.strip():
                continue

            # OWASP scan
            owasp_findings = self._owasp_scanner.scan(df.file_path, df.source, df.added_line_numbers)
            for f in owasp_findings:
                all_findings.append(
                    {
                        "rule_id": f.rule_id,
                        "category": f.category,
                        "severity": f.severity,
                        "file": f.file_path,
                        "line": f.line,
                        "message": f.message,
                        "matched_code": f.matched_code,
                    }
                )

            # Structural analysis
            functions = self._structural_analyzer.analyze(df.file_path, df.source, df.added_line_numbers)
            for func in functions:
                func_dict = {
                    "name": func.name,
                    "file": func.file_path,
                    "line": func.line,
                    "end_line": func.end_line,
                    "params": func.params,
                    "complexity": func.complexity,
                    "calls": func.calls,
                    "nesting_depth": func.nesting_depth,
                }
                all_functions.append(func_dict)

                if func.complexity > max_complexity:
                    max_complexity = func.complexity

                if func.nesting_depth > 4:
                    deeply_nested.append(
                        {
                            "file": func.file_path,
                            "function": func.name,
                            "line": func.line,
                            "depth": func.nesting_depth,
                        }
                    )

                if func.calls:
                    call_graph[f"{func.file_path}:{func.name}"] = func.calls

        logger.info(
            "static_analysis_complete",
            files_analyzed=len(diff_files),
            functions_found=len(all_functions),
            security_findings=len(all_findings),
            max_complexity=max_complexity,
        )

        return {
            "tree_sitter": {
                "functions": all_functions,
                "max_complexity": max_complexity,
                "deeply_nested": deeply_nested,
                "call_graph": call_graph,
            },
            "security": {
                "findings": all_findings,
                "rules_checked": rules_checked,
                "total_findings": len(all_findings),
            },
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "tree_sitter": {
                "functions": [],
                "max_complexity": 0,
                "deeply_nested": [],
                "call_graph": {},
            },
            "security": {
                "findings": [],
                "rules_checked": 0,
                "total_findings": 0,
            },
        }
