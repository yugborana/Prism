"""
Prism Diff Parser — Extracts valid diff line ranges for inline comments.

GitHub only allows inline comments on lines that appear in the diff.
This module parses the unified diff format to build a set of valid
(file_path, line_number) pairs that can be targeted by inline comments.

It also provides an "annotated diff" that numbers each line clearly,
making it easier for the LLM to reference exact line numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from observability.logging import get_logger

logger = get_logger(__name__)

# Matches unified diff hunk headers: @@ -old_start,old_count +new_start,new_count @@
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

# Matches diff file headers: diff --git a/path b/path  OR  +++ b/path
_FILE_RE = re.compile(r"^\+\+\+ b/(.+)")

# Matches the start of a new file in a unified diff
_DIFF_HEADER_RE = re.compile(r"^diff --git a/")


def split_diff_by_file(diff_text: str) -> dict[str, str]:
    """Split a unified diff into per-file diffs.

    Returns a dict mapping file_path -> that file's complete diff section
    (including the diff --git header, ---, +++, and all hunks).
    """
    files: dict[str, str] = {}
    current_file: str | None = None
    current_lines: list[str] = []

    for line in diff_text.split("\n"):
        if _DIFF_HEADER_RE.match(line):
            # Save previous file's diff
            if current_file and current_lines:
                files[current_file] = "\n".join(current_lines)
            current_lines = [line]
            current_file = None  # Will be set by +++ line
        else:
            current_lines.append(line)
            # Detect file path from +++ line
            file_match = _FILE_RE.match(line)
            if file_match:
                current_file = file_match.group(1)

    # Save last file
    if current_file and current_lines:
        files[current_file] = "\n".join(current_lines)

    return files


@dataclass
class DiffFileInfo:
    """Parsed diff info for a single file."""

    file_path: str
    valid_lines: set[int] = field(default_factory=set)  # Lines that appear in the diff
    added_lines: set[int] = field(default_factory=set)  # Lines that are additions (+)
    hunks: list[tuple[int, int]] = field(default_factory=list)  # (start, end) of each hunk


def parse_diff_valid_lines(diff_text: str) -> dict[str, DiffFileInfo]:
    """Parse a unified diff and return valid line ranges per file.

    A "valid line" is any line that appears in a diff hunk — either a
    context line (unchanged), an added line (+), or a deleted line (-).
    GitHub allows inline comments on all of these.

    Returns:
        Dict mapping file_path -> DiffFileInfo with valid line numbers.
    """
    result: dict[str, DiffFileInfo] = {}
    current_file: str | None = None
    current_line = 0

    for raw_line in diff_text.split("\n"):
        # Detect file boundary
        file_match = _FILE_RE.match(raw_line)
        if file_match:
            current_file = file_match.group(1)
            if current_file not in result:
                result[current_file] = DiffFileInfo(file_path=current_file)
            continue

        if current_file is None:
            continue

        info = result[current_file]

        # Detect hunk header
        hunk_match = _HUNK_RE.match(raw_line)
        if hunk_match:
            current_line = int(hunk_match.group(1))
            continue

        # Lines inside a hunk
        if raw_line.startswith("+"):
            # Added line — valid for comments at current_line
            info.valid_lines.add(current_line)
            info.added_lines.add(current_line)
            current_line += 1
        elif raw_line.startswith("-"):
            # Deleted line — no new-file line number exists, so it cannot
            # be targeted by inline comments using the `line` field.
            # (Would need `side: "LEFT"` + `original_line` for multi-line API)
            pass
        elif raw_line.startswith(" "):
            # Context line — valid for comments
            info.valid_lines.add(current_line)
            current_line += 1
        # else: diff metadata lines (e.g., "\ No newline at end of file")

    # Build hunk ranges from valid_lines
    for file_path, info in result.items():
        if info.valid_lines:
            sorted_lines = sorted(info.valid_lines)
            hunk_start = sorted_lines[0]
            prev = sorted_lines[0]
            for line in sorted_lines[1:]:
                if line > prev + 1:
                    info.hunks.append((hunk_start, prev))
                    hunk_start = line
                prev = line
            info.hunks.append((hunk_start, prev))

    return result


def build_annotated_diff(diff_text: str) -> str:
    """Build a line-numbered version of the diff for LLM context.

    Each line in the diff gets a clear marker showing:
    - The file it belongs to
    - The exact line number in the new version of the file
    - Whether it's an addition (+), deletion (-), or context

    This makes it impossible for the LLM to misattribute line numbers.
    """
    parts: list[str] = []
    current_file: str | None = None
    current_line = 0

    for raw_line in diff_text.split("\n"):
        file_match = _FILE_RE.match(raw_line)
        if file_match:
            current_file = file_match.group(1)
            parts.append(f"\n{'=' * 60}")
            parts.append(f"FILE: {current_file}")
            parts.append(f"{'=' * 60}")
            continue

        if current_file is None:
            # Skip diff headers before first file
            if raw_line.startswith("diff --git") or raw_line.startswith("---"):
                continue
            parts.append(raw_line)
            continue

        hunk_match = _HUNK_RE.match(raw_line)
        if hunk_match:
            current_line = int(hunk_match.group(1))
            parts.append(f"\n--- hunk starting at line {current_line} ---")
            continue

        if raw_line.startswith("+"):
            parts.append(f"  L{current_line:>4} [+] {raw_line[1:]}")
            current_line += 1
        elif raw_line.startswith("-"):
            parts.append(f"       [-] {raw_line[1:]}")
            # Deleted lines don't have a new-file line number
        elif raw_line.startswith(" "):
            parts.append(f"  L{current_line:>4}     {raw_line[1:]}")
            current_line += 1
        elif raw_line.startswith("\\"):
            continue  # "\ No newline at end of file"

    return "\n".join(parts)


def filter_valid_comments(
    inline_comments: list[dict],
    diff_info: dict[str, DiffFileInfo],
) -> tuple[list[dict], list[dict]]:
    """Filter inline comments to only include those on valid diff lines.

    Returns:
        Tuple of (valid_comments, rejected_comments).
        Rejected comments are logged but not posted inline — they'll be
        included in the summary instead.
    """
    valid = []
    rejected = []

    for comment in inline_comments:
        path = comment.get("path", "")
        line = comment.get("line", 0)

        file_info = diff_info.get(path)
        if file_info and line in file_info.valid_lines:
            valid.append(comment)
        else:
            rejected.append(comment)
            logger.debug(
                "inline_comment_rejected",
                path=path,
                line=line,
                reason="line_not_in_diff",
                valid_range=str(file_info.hunks) if file_info else "file_not_in_diff",
            )

    if rejected:
        logger.info(
            "inline_comments_filtered",
            total=len(inline_comments),
            valid=len(valid),
            rejected=len(rejected),
        )

    return valid, rejected
