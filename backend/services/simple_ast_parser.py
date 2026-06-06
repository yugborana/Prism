from typing import Any, Dict, List, Tuple

import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

# Language modules and mapping
LANGUAGE_MODULES = {
    "python": tspython,
    "javascript": tsjs,
    "typescript": tsts,
    "go": tsgo,
    "rust": tsrust,
}

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
}


class SimpleASTParser:
    """
    Simplified AST parser focused on core functionality only.
    Removes all the complex analysis from the original.
    """

    def __init__(self, language: str = "python"):
        if language not in LANGUAGE_MODULES:
            raise ValueError(f"Unsupported language: {language}")

        lang_module = LANGUAGE_MODULES[language]
        self.language = Language(lang_module.language())
        self.parser = Parser(self.language)
        self.lang_name = language

    def parse_file(self, file_path: str) -> Tuple[Any, str]:
        with open(file_path, "rb") as f:
            source_code = f.read()

        tree = self.parser.parse(source_code)
        if isinstance(source_code, bytes):
            source_code = source_code.decode("utf-8")

        return tree, source_code

    def extract_functions(self, tree, source_code: str) -> List[Dict[str, Any]]:
        functions = []

        def node_text(node):
            try:
                return source_code[node.start_byte : node.end_byte]
            except Exception:
                return ""

        def walk(node):
            if self.lang_name == "python" and node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    params_node = node.child_by_field_name("parameters")
                    params = node_text(params_node) if params_node else None

                    functions.append(
                        {
                            "name": name,
                            "type": "function",
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "parameters": params,
                            "signature": node_text(node),
                            "source": node_text(node),
                        }
                    )

            elif (
                self.lang_name in ("javascript", "typescript")
                and node.type == "function_declaration"
            ):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    params_node = node.child_by_field_name("parameters")
                    params = node_text(params_node) if params_node else None

                    functions.append(
                        {
                            "name": name,
                            "type": "function",
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "parameters": params,
                            "signature": node_text(node),
                            "source": node_text(node),
                        }
                    )

            elif self.lang_name == "go" and node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    params_node = node.child_by_field_name("parameters")
                    params = node_text(params_node) if params_node else None

                    functions.append(
                        {
                            "name": name,
                            "type": "function",
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "parameters": params,
                            "signature": node_text(node),
                            "source": node_text(node),
                        }
                    )

            elif self.lang_name == "go" and node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    params_node = node.child_by_field_name("parameters")
                    receiver_node = node.child_by_field_name("receiver")
                    params = node_text(params_node) if params_node else None
                    receiver = node_text(receiver_node) if receiver_node else None

                    functions.append(
                        {
                            "name": name,
                            "type": "method",
                            "receiver": receiver,
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "parameters": params,
                            "signature": node_text(node),
                            "source": node_text(node),
                        }
                    )

            elif self.lang_name == "rust" and node.type == "function_item":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    params_node = node.child_by_field_name("parameters")
                    params = node_text(params_node) if params_node else None

                    functions.append(
                        {
                            "name": name,
                            "type": "function",
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "parameters": params,
                            "signature": node_text(node),
                            "source": node_text(node),
                        }
                    )

            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return functions

    def extract_classes(self, tree, source_code: str) -> List[Dict[str, Any]]:
        classes = []

        def node_text(node):
            try:
                return source_code[node.start_byte : node.end_byte]
            except Exception:
                return ""

        def walk(node):
            if self.lang_name == "python" and node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    classes.append(
                        {
                            "name": name,
                            "type": "class",
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "source": node_text(node),
                        }
                    )

            elif (
                self.lang_name in ("javascript", "typescript")
                and node.type == "class_declaration"
            ):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    classes.append(
                        {
                            "name": name,
                            "type": "class",
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "source": node_text(node),
                        }
                    )

            elif self.lang_name == "go" and node.type == "type_spec":
                name_node = node.child_by_field_name("name")
                type_node = node.child_by_field_name("type")
                if name_node and type_node:
                    name = node_text(name_node)
                    type_kind = type_node.type

                    # Determine if it's struct or interface
                    if type_kind == "struct_type":
                        classes.append(
                            {
                                "name": name,
                                "type": "struct",
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                                "source": node_text(node),
                            }
                        )
                    elif type_kind == "interface_type":
                        classes.append(
                            {
                                "name": name,
                                "type": "interface",
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                                "source": node_text(node),
                            }
                        )

            elif self.lang_name == "rust" and node.type == "struct_item":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    classes.append(
                        {
                            "name": name,
                            "type": "struct",
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "source": node_text(node),
                        }
                    )

            elif self.lang_name == "rust" and node.type == "impl_item":
                type_node = node.child_by_field_name("type")
                trait_node = node.child_by_field_name("trait")
                if type_node:
                    name = node_text(type_node)
                    impl_type = "impl"
                    if trait_node:
                        trait_name = node_text(trait_node)
                        name = f"{trait_name} for {name}"
                        impl_type = "trait_impl"

                    classes.append(
                        {
                            "name": name,
                            "type": impl_type,
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "source": node_text(node),
                        }
                    )

            elif self.lang_name == "rust" and node.type == "trait_item":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = node_text(name_node)
                    classes.append(
                        {
                            "name": name,
                            "type": "trait",
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "source": node_text(node),
                        }
                    )

            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return classes

    def extract_imports(self, tree, source_code: str) -> List[str]:
        imports = []

        def node_text(node):
            try:
                return source_code[node.start_byte : node.end_byte]
            except Exception:
                return ""

        def walk(node):
            if self.lang_name == "python":
                if node.type == "import_from_statement":
                    for child in node.children:
                        if child.type == "dotted_name":
                            module = node_text(child)
                            imports.append(module)
                            break
                elif node.type == "import_statement":
                    for child in node.children:
                        if child.type == "dotted_name":
                            module = node_text(child)
                            imports.append(module)

            elif self.lang_name in ("javascript", "typescript"):
                if node.type == "import_statement":
                    for child in node.children:
                        if child.type == "string":
                            text = node_text(child).strip('"').strip("'")
                            imports.append(text)

            elif self.lang_name == "go":
                if node.type == "import_spec":
                    # Import spec contains a string with the package path
                    for child in node.children:
                        if child.type == "interpreted_string_literal":
                            text = node_text(child).strip('"')
                            imports.append(text)

            elif self.lang_name == "rust":
                if node.type == "use_declaration":
                    # Extract the argument field which contains the import path
                    arg_node = node.child_by_field_name("argument")
                    if arg_node:
                        import_path = node_text(arg_node)
                        imports.append(import_path)

            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return imports

    def extract_semantic_analysis(
        self, tree, source_code: str, file_path: str
    ) -> Dict[str, Any]:
        return {
            "file_path": file_path,
            "language": self.lang_name,
            "functions": self.extract_functions(tree, source_code),
            "classes": self.extract_classes(tree, source_code),
            "imports": self.extract_imports(tree, source_code),
            "analysis_method": "simplified_ast",
        }
