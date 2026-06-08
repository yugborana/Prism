"""
Prism Merkle Tree — Incremental Change Detection for Repo Indexing.

Inspired by Cursor's secure codebase indexing approach:
  https://cursor.com/blog/secure-codebase-indexing

Builds a cryptographic hash tree where:
  - Each leaf = SHA-256 of a source file's contents
  - Each internal node = SHA-256 of its children's hashes concatenated (sorted)
  - Root hash = single value representing the entire repo state

Diffing two trees is O(changed files), not O(total files):
  - Compare root hashes → if equal, entire repo unchanged
  - If different, recurse into children
  - If a subtree's hash matches, skip it entirely (all files unchanged)

Also derives a SimHash (locality-sensitive hash) for cross-repo similarity
detection, enabling index reuse across forks/templates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from observability.logging import get_logger

logger = get_logger(__name__)

# File extensions we index (must match SyntacticChunker's supported languages)
SUPPORTED_EXTENSIONS: set[str] = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs"}

# Directories to always skip during tree construction
SKIP_DIRS: set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    "target",
    "vendor",
    ".eggs",
    "*.egg-info",
}


@dataclass
class MerkleNode:
    """A node in the Merkle tree — either a file (leaf) or directory (internal)."""

    path: str  # Relative path from repo root (e.g., "src/services/billing.py")
    hash: str  # SHA-256 hex digest
    is_dir: bool
    children: dict[str, "MerkleNode"] = field(default_factory=dict)

    @property
    def child_count(self) -> int:
        """Total number of leaf nodes (files) in this subtree."""
        if not self.is_dir:
            return 1
        return sum(c.child_count for c in self.children.values())


class MerkleTree:
    """
    Merkle hash tree for a repository.

    Usage::

        # Build from a cloned repo directory
        tree = MerkleTree.build_from_directory(Path("/tmp/repos/acme/backend"))

        # Compare two snapshots — returns only changed file paths
        changed = old_tree.diff(new_tree)  # ["src/billing.py", "tests/test_billing.py"]

        # Compute a locality-sensitive hash for cross-repo similarity
        sim = tree.simhash()  # 256-bit SimHash as hex string
    """

    def __init__(self, root: MerkleNode):
        self.root = root

    @classmethod
    def build_from_directory(
        cls,
        repo_path: Path,
        extensions: set[str] | None = None,
    ) -> "MerkleTree":
        """Walk the repo, hash every supported file, build the tree bottom-up.

        Args:
            repo_path: Absolute path to the cloned repository.
            extensions: File extensions to include. Defaults to SUPPORTED_EXTENSIONS.

        Returns:
            A fully constructed MerkleTree.
        """
        if extensions is None:
            extensions = SUPPORTED_EXTENSIONS

        root_node = cls._build_node(repo_path, repo_path, extensions)
        logger.info(
            "merkle_tree_built",
            root_hash=root_node.hash[:12],
            total_files=root_node.child_count,
        )
        return cls(root_node)

    @classmethod
    def _build_node(cls, current_path: Path, repo_root: Path, extensions: set[str]) -> MerkleNode:
        """Recursively build a MerkleNode for a file or directory."""
        relative = str(current_path.relative_to(repo_root)).replace("\\", "/")
        if relative == ".":
            relative = ""

        if current_path.is_file():
            # Leaf node: hash the file contents
            file_hash = cls._hash_file(current_path)
            return MerkleNode(path=relative, hash=file_hash, is_dir=False)

        # Directory node: recurse into children, then hash the sorted child hashes
        children: dict[str, MerkleNode] = {}
        try:
            entries = sorted(current_path.iterdir(), key=lambda p: p.name)
        except PermissionError:
            entries = []

        for entry in entries:
            # Skip excluded directories
            if entry.is_dir() and entry.name in SKIP_DIRS:
                continue
            # Skip hidden files/dirs (except .github, etc.)
            if entry.name.startswith(".") and entry.name not in {".github"}:
                continue

            if entry.is_dir():
                child = cls._build_node(entry, repo_root, extensions)
                # Only include directories that contain supported files
                if child.child_count > 0:
                    children[entry.name] = child
            elif entry.suffix in extensions:
                child = cls._build_node(entry, repo_root, extensions)
                children[entry.name] = child

        # Internal node hash: SHA-256 of sorted "name:hash" pairs
        if children:
            hasher = hashlib.sha256()
            for name in sorted(children.keys()):
                hasher.update(f"{name}:{children[name].hash}\n".encode())
            dir_hash = hasher.hexdigest()
        else:
            dir_hash = hashlib.sha256(b"empty").hexdigest()

        return MerkleNode(path=relative, hash=dir_hash, is_dir=True, children=children)

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """SHA-256 hash of a file's contents. Reads in chunks for large files."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except (OSError, PermissionError):
            # If we can't read the file, use a sentinel hash
            hasher.update(b"unreadable")
        return hasher.hexdigest()

    def root_hash(self) -> str:
        """The root hash — a single string representing the entire repo state."""
        return self.root.hash

    def diff(self, other: "MerkleTree") -> list[str]:
        """Compare two Merkle trees, return list of changed file paths.

        Walks both trees simultaneously. When a directory node's hash matches,
        the entire subtree is skipped. Only divergent branches are explored.

        Returns:
            List of relative file paths that differ between the two trees.
            Includes files added, modified, or deleted.
        """
        changed: list[str] = []
        self._diff_nodes(self.root, other.root, changed)
        return changed

    @classmethod
    def _diff_nodes(
        cls,
        old: MerkleNode | None,
        new: MerkleNode | None,
        changed: list[str],
    ) -> None:
        """Recursively compare two nodes and collect changed file paths."""
        # File deleted
        if old is not None and new is None:
            if old.is_dir:
                cls._collect_all_files(old, changed)
            else:
                changed.append(old.path)
            return

        # File added
        if old is None and new is not None:
            if new.is_dir:
                cls._collect_all_files(new, changed)
            else:
                changed.append(new.path)
            return

        # Both exist
        assert old is not None and new is not None

        # Hashes match → entire subtree unchanged, skip
        if old.hash == new.hash:
            return

        # Both are files with different hashes → file changed
        if not old.is_dir and not new.is_dir:
            changed.append(new.path)
            return

        # Type mismatch (dir→file or file→dir) — treat as delete + add
        if old.is_dir != new.is_dir:
            if old.is_dir:
                cls._collect_all_files(old, changed)
            else:
                changed.append(old.path)
            if new.is_dir:
                cls._collect_all_files(new, changed)
            else:
                changed.append(new.path)
            return

        # Both are directories with different hashes → recurse into children
        all_names = set(old.children.keys()) | set(new.children.keys())
        for name in sorted(all_names):
            old_child = old.children.get(name)
            new_child = new.children.get(name)
            cls._diff_nodes(old_child, new_child, changed)

    @staticmethod
    def _collect_all_files(node: MerkleNode, result: list[str]) -> None:
        """Collect all leaf file paths under a node."""
        if not node.is_dir:
            result.append(node.path)
            return
        for child in node.children.values():
            MerkleTree._collect_all_files(child, result)

    def get_all_files(self) -> list[str]:
        """Return all file paths in the tree."""
        files: list[str] = []
        self._collect_all_files(self.root, files)
        return files

    # ── SimHash ───────────────────────────────────────────────────────────

    def simhash(self) -> str:
        """Compute a 256-bit locality-sensitive hash (SimHash) from all leaf hashes.

        Two repos sharing 90% of files will have SimHash Hamming distance < 25.

        Algorithm:
          1. For each leaf file hash (treated as 256-bit feature):
             - If bit i is 1: counter[i] += 1
             - If bit i is 0: counter[i] -= 1
          2. Final SimHash: bit i = 1 if counter[i] > 0, else 0

        Returns:
            64-char hex string (256 bits).
        """
        counters = [0] * 256
        file_count = 0

        for file_hash_hex in self._iter_leaf_hashes(self.root):
            file_count += 1
            hash_bytes = bytes.fromhex(file_hash_hex)
            for byte_idx, byte_val in enumerate(hash_bytes):
                for bit_idx in range(8):
                    global_bit = byte_idx * 8 + bit_idx
                    if global_bit >= 256:
                        break
                    if byte_val & (1 << (7 - bit_idx)):
                        counters[global_bit] += 1
                    else:
                        counters[global_bit] -= 1

        # Build the final SimHash
        result_bytes = bytearray(32)
        for i in range(256):
            if counters[i] > 0:
                result_bytes[i // 8] |= 1 << (7 - (i % 8))

        return result_bytes.hex()

    @staticmethod
    def _iter_leaf_hashes(node: MerkleNode):
        """Yield all leaf (file) hashes in the tree."""
        if not node.is_dir:
            yield node.hash
            return
        for child in node.children.values():
            yield from MerkleTree._iter_leaf_hashes(child)

    @staticmethod
    def hamming_distance(hash_a: str, hash_b: str) -> int:
        """Compute Hamming distance between two SimHash hex strings.

        Each bit position that differs counts as 1.
        Lower distance = more similar repos.
        Threshold: < 25 out of 256 bits = >90% file overlap.
        """
        bytes_a = bytes.fromhex(hash_a)
        bytes_b = bytes.fromhex(hash_b)
        distance = 0
        for ba, bb in zip(bytes_a, bytes_b):
            xor = ba ^ bb
            distance += bin(xor).count("1")
        return distance

    # ── Serialization ─────────────────────────────────────────────────────

    def serialize(self) -> dict[str, Any]:
        """Serialize the tree to a JSON-compatible dict for Redis storage."""
        return self._serialize_node(self.root)

    @classmethod
    def _serialize_node(cls, node: MerkleNode) -> dict[str, Any]:
        result: dict[str, Any] = {
            "p": node.path,
            "h": node.hash,
            "d": node.is_dir,
        }
        if node.is_dir and node.children:
            result["c"] = {name: cls._serialize_node(child) for name, child in node.children.items()}
        return result

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "MerkleTree":
        """Deserialize from a JSON-compatible dict."""
        root = cls._deserialize_node(data)
        return cls(root)

    @classmethod
    def _deserialize_node(cls, data: dict[str, Any]) -> MerkleNode:
        children: dict[str, MerkleNode] = {}
        if "c" in data:
            children = {name: cls._deserialize_node(child_data) for name, child_data in data["c"].items()}
        return MerkleNode(
            path=data["p"],
            hash=data["h"],
            is_dir=data["d"],
            children=children,
        )
