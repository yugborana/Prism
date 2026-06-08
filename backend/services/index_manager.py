"""
Prism Index Manager — Orchestrates the Full Repo Indexing Pipeline.

Coordinates: clone → Merkle tree → SimHash check → chunk → embed → upsert.

Two modes:
  1. Full index: For a never-before-seen repo (or SimHash copy + diff update)
  2. Incremental refresh: For an already-indexed repo (Merkle diff → re-chunk changed files)

Implements a 30-day TTL: repo indexes not queried in 30 days are deleted
from Qdrant to manage storage growth.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from observability.logging import get_logger
from services.embedding_cache import EmbeddingCache
from services.merkle_tree import MerkleTree
from services.repo_cloner import RepoCloner
from services.simhash import SimHashIndex
from services.syntactic_chunker import CodeChunk, SyntacticChunker

logger = get_logger(__name__)

# Max files to index — skip very large monorepos
MAX_FILES_TO_INDEX = 5000
# Embedding batch size
EMBED_BATCH_SIZE = 50
# Index TTL: 30 days without a query → delete
INDEX_TTL_DAYS = 30


@dataclass
class IndexStats:
    """Statistics from an indexing run."""

    repo_name: str
    total_files: int = 0
    files_changed: int = 0
    total_chunks: int = 0
    chunks_embedded: int = 0      # Actually called embedding API
    chunks_cached: int = 0        # Hit embedding cache
    chunks_copied: int = 0        # Copied from SimHash match
    duration_seconds: float = 0.0
    simhash_source: str | None = None  # Repo we copied from, if any
    skipped_reason: str | None = None  # If indexing was skipped, why


class IndexManager:
    """Orchestrates the entire repo indexing pipeline.

    Usage::

        manager = IndexManager()

        # Check if index exists
        status = await manager.check_index_status("acme/backend")

        # Full index (background task)
        stats = await manager.build_full_index("acme/backend", "main", 12345)

        # Incremental refresh (background task)
        stats = await manager.refresh_index("acme/backend", "main", 12345)
    """

    def __init__(self):
        self.cloner = RepoCloner()
        self.chunker = SyntacticChunker()
        self.cache = EmbeddingCache()
        self.simhash = SimHashIndex()

    async def check_index_status(self, repo_name: str) -> str:
        """Returns 'fresh' | 'stale' | 'none'."""
        return await self.simhash.check_index_status(repo_name)

    async def build_full_index(
        self,
        repo_name: str,
        base_branch: str,
        installation_id: int,
    ) -> IndexStats:
        """Full indexing pipeline for a never-before-seen repo.

        1. Check SimHash for a reusable index within same installation → copy if found
        2. Clone repo (shallow)
        3. Build Merkle tree from disk
        4. Chunk all supported files with tree-sitter
        5. Embed chunks (with cache) → upsert to Qdrant 'repo_chunks'
        6. Store Merkle tree + SimHash in Redis
        """
        start_time = time.monotonic()
        stats = IndexStats(repo_name=repo_name)

        try:
            # Step 1: Clone
            repo_path = await self.cloner.ensure_clone(
                repo_name, base_branch, installation_id
            )

            # Step 2: Build Merkle tree
            merkle = await asyncio.to_thread(
                MerkleTree.build_from_directory, repo_path
            )
            all_files = merkle.get_all_files()
            stats.total_files = len(all_files)

            # Guard: skip very large repos
            if stats.total_files > MAX_FILES_TO_INDEX:
                stats.skipped_reason = f"Too many files ({stats.total_files} > {MAX_FILES_TO_INDEX})"
                logger.warning(
                    "index_skipped_too_large",
                    repo=repo_name,
                    file_count=stats.total_files,
                )
                return stats

            # Step 3: Check SimHash for reusable index (same installation only)
            similar_repo = await self.simhash.find_similar(
                merkle, installation_id, exclude_repo=repo_name
            )

            if similar_repo:
                # Copy the existing index
                copied = await self.simhash.copy_index(similar_repo, repo_name)
                stats.chunks_copied = copied
                stats.simhash_source = similar_repo

                # Now do a Merkle diff against the source to find what changed
                source_merkle = await self.simhash.get_stored_merkle_tree(similar_repo)
                if source_merkle:
                    changed_files = merkle.diff(source_merkle)
                    stats.files_changed = len(changed_files)
                else:
                    # Can't diff — just re-index everything
                    changed_files = all_files
                    stats.files_changed = len(all_files)

                logger.info(
                    "index_simhash_reuse",
                    repo=repo_name,
                    source=similar_repo,
                    copied=copied,
                    files_to_update=len(changed_files),
                )
            else:
                # No similar repo — index all files
                changed_files = all_files
                stats.files_changed = len(all_files)

            # Step 4: Chunk the files that need indexing
            chunks = await self._chunk_files(repo_path, changed_files)
            stats.total_chunks = len(chunks)

            # Step 5: Embed + upsert (if there are new chunks)
            if chunks:
                # Delete old chunks for changed files before upserting new ones
                if similar_repo:
                    await self._delete_chunks_for_files(
                        repo_name, [c.file_path for c in chunks]
                    )

                embedded, cached = await self._embed_and_upsert(
                    chunks, repo_name
                )
                stats.chunks_embedded = embedded
                stats.chunks_cached = cached

            # Step 6: Store Merkle tree + SimHash
            await self.simhash.store(repo_name, merkle, installation_id)
            await self.simhash.store_merkle_tree(repo_name, merkle)

        except Exception as e:
            logger.error(
                "index_build_failed",
                repo=repo_name,
                error=str(e),
                exc_info=True,
            )
            stats.skipped_reason = str(e)

        stats.duration_seconds = time.monotonic() - start_time
        logger.info(
            "index_build_complete",
            repo=repo_name,
            duration=f"{stats.duration_seconds:.1f}s",
            files=stats.total_files,
            chunks=stats.total_chunks,
            embedded=stats.chunks_embedded,
            cached=stats.chunks_cached,
            copied=stats.chunks_copied,
            source=stats.simhash_source,
        )
        return stats

    async def refresh_index(
        self,
        repo_name: str,
        base_branch: str,
        installation_id: int,
    ) -> IndexStats:
        """Incremental update for an already-indexed repo.

        1. git fetch → update clone
        2. Build new Merkle tree
        3. Diff against stored Merkle tree → get list of changed files
        4. Re-chunk only changed files
        5. Embed new chunks (cache handles unchanged ones)
        6. Delete old chunks for changed files from Qdrant, upsert new
        7. Update stored Merkle tree + SimHash
        """
        start_time = time.monotonic()
        stats = IndexStats(repo_name=repo_name)

        try:
            # Step 1: Update clone
            repo_path = await self.cloner.ensure_clone(
                repo_name, base_branch, installation_id
            )

            # Step 2: Build new Merkle tree
            new_merkle = await asyncio.to_thread(
                MerkleTree.build_from_directory, repo_path
            )
            stats.total_files = len(new_merkle.get_all_files())

            # Step 3: Diff against stored tree
            old_merkle = await self.simhash.get_stored_merkle_tree(repo_name)
            if old_merkle:
                changed_files = new_merkle.diff(old_merkle)
            else:
                # No stored tree — treat as full re-index
                changed_files = new_merkle.get_all_files()

            stats.files_changed = len(changed_files)

            if not changed_files:
                logger.info("index_no_changes", repo=repo_name)
                stats.duration_seconds = time.monotonic() - start_time
                return stats

            # Step 4: Chunk changed files
            chunks = await self._chunk_files(repo_path, changed_files)
            stats.total_chunks = len(chunks)

            # Step 5: Delete old chunks for changed files, upsert new
            if chunks:
                unique_files = list({c.file_path for c in chunks})
                await self._delete_chunks_for_files(repo_name, unique_files)

                embedded, cached = await self._embed_and_upsert(
                    chunks, repo_name
                )
                stats.chunks_embedded = embedded
                stats.chunks_cached = cached

            # Step 6: Update stored Merkle + SimHash
            await self.simhash.store(repo_name, new_merkle, installation_id)
            await self.simhash.store_merkle_tree(repo_name, new_merkle)

        except Exception as e:
            logger.error(
                "index_refresh_failed",
                repo=repo_name,
                error=str(e),
                exc_info=True,
            )
            stats.skipped_reason = str(e)

        stats.duration_seconds = time.monotonic() - start_time
        logger.info(
            "index_refresh_complete",
            repo=repo_name,
            duration=f"{stats.duration_seconds:.1f}s",
            files_changed=stats.files_changed,
            chunks=stats.total_chunks,
            embedded=stats.chunks_embedded,
            cached=stats.chunks_cached,
        )
        return stats

    async def cleanup_stale_indexes(self, max_age_days: int = INDEX_TTL_DAYS) -> int:
        """Delete repo indexes not updated in max_age_days from Qdrant.

        Called by a Celery beat task (daily). Manages storage growth.
        Returns count of repos cleaned.
        """
        from utils.connections import get_redis

        redis_client = get_redis()
        if redis_client is None:
            logger.warning("cleanup_stale_indexes_redis_unavailable")
            return 0

        cleaned = 0
        cutoff = time.time() - (max_age_days * 86400)

        try:
            # Scan all index metadata keys
            async for key in redis_client.scan_iter(match="prism:index_meta:*"):
                try:
                    meta_raw = await redis_client.get(key)
                    if not meta_raw:
                        continue

                    import json
                    meta = json.loads(meta_raw)
                    indexed_at = meta.get("indexed_at", 0)

                    if indexed_at < cutoff:
                        repo_name = key.replace("prism:index_meta:", "")
                        installation_id = meta.get("installation_id", 0)

                        # Delete from Qdrant
                        await self._delete_all_repo_chunks(repo_name)

                        # Delete metadata from Redis
                        await self.simhash.delete_index_meta(
                            repo_name, installation_id
                        )

                        cleaned += 1
                        logger.info(
                            "index_ttl_cleanup",
                            repo=repo_name,
                            age_days=int((time.time() - indexed_at) / 86400),
                        )
                except Exception as e:
                    logger.warning("index_ttl_cleanup_error", key=key, error=str(e))

        except Exception as e:
            logger.error("index_ttl_scan_failed", error=str(e))

        if cleaned:
            logger.info("index_ttl_cleanup_complete", repos_cleaned=cleaned)

        return cleaned

    # ── Internal Helpers ──────────────────────────────────────────────────

    async def _chunk_files(
        self, repo_path: Path, file_paths: list[str]
    ) -> list[CodeChunk]:
        """Chunk multiple files using the syntactic chunker."""
        all_chunks: list[CodeChunk] = []

        for file_path in file_paths:
            try:
                chunks = await asyncio.to_thread(
                    self.chunker.chunk_file_from_disk, repo_path, file_path
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(
                    "chunk_file_failed",
                    file=file_path,
                    error=str(e),
                )

        return all_chunks

    async def _embed_and_upsert(
        self, chunks: list[CodeChunk], repo_name: str
    ) -> tuple[int, int]:
        """Embed chunks (with cache) and upsert to Qdrant.

        Returns (chunks_embedded, chunks_cached).
        """
        from qdrant_client.models import PointStruct
        from utils.llm_factory import LLMClient
        from utils.qdrant_client import get_qdrant_client

        llm = LLMClient()
        client = get_qdrant_client()

        total_embedded = 0
        total_cached = 0

        # Process in batches
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]

            # Prepare embedding inputs
            content_hashes = [c.content_hash for c in batch]
            texts = [c.to_embedding_text() for c in batch]

            # Get embeddings (from cache or freshly generated)
            cache_stats_before = self.cache.stats()
            embeddings = await self.cache.get_or_embed(
                content_hashes,
                llm.embed_batch,
                texts,
            )
            cache_stats_after = self.cache.stats()

            batch_cached = cache_stats_after["hits"] - cache_stats_before["hits"]
            batch_embedded = len(batch) - batch_cached
            total_embedded += batch_embedded
            total_cached += batch_cached

            # Build Qdrant points
            points = []
            for chunk, embedding in zip(batch, embeddings):
                payload = chunk.to_payload()
                payload["repo_name"] = repo_name
                payload["indexed_at"] = time.time()

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=payload,
                    )
                )

            # Upsert to Qdrant
            if points:
                try:
                    await asyncio.to_thread(
                        client.upsert,
                        collection_name="repo_chunks",
                        points=points,
                    )
                except Exception as e:
                    logger.error(
                        "qdrant_upsert_failed",
                        repo=repo_name,
                        batch_size=len(points),
                        error=str(e),
                    )

        return total_embedded, total_cached

    async def _delete_chunks_for_files(
        self, repo_name: str, file_paths: list[str]
    ) -> None:
        """Delete existing Qdrant points for specific files in a repo."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from utils.qdrant_client import get_qdrant_client

        client = get_qdrant_client()

        for file_path in file_paths:
            try:
                await asyncio.to_thread(
                    client.delete,
                    collection_name="repo_chunks",
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="repo_name",
                                match=MatchValue(value=repo_name),
                            ),
                            FieldCondition(
                                key="file_path",
                                match=MatchValue(value=file_path),
                            ),
                        ]
                    ),
                )
            except Exception as e:
                logger.warning(
                    "chunk_delete_failed",
                    repo=repo_name,
                    file=file_path,
                    error=str(e),
                )

    async def _delete_all_repo_chunks(self, repo_name: str) -> None:
        """Delete ALL Qdrant points for a repo (TTL cleanup)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from utils.qdrant_client import get_qdrant_client

        client = get_qdrant_client()
        try:
            await asyncio.to_thread(
                client.delete,
                collection_name="repo_chunks",
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="repo_name",
                            match=MatchValue(value=repo_name),
                        ),
                    ]
                ),
            )
        except Exception as e:
            logger.warning(
                "repo_chunks_delete_failed",
                repo=repo_name,
                error=str(e),
            )

    async def close(self) -> None:
        """Cleanup resources."""
        await self.cache.close()
        await self.simhash.close()
