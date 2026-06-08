"""
Prism SimHash Index — Cross-Repo Index Reuse via Locality-Sensitive Hashing.

When a new repo's first PR arrives, instead of building a full index from
scratch (~25s), check if a similar repo is already indexed. If
"acme/backend-v2" is a fork of "acme/backend", 95% of the chunks are
identical. Copy the index, diff-update only the 5% that changed.

Restriction: Index reuse only works within the same GitHub App installation
(same organization). A review for org-a can never see or reuse indexes from org-b.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from observability.logging import get_logger
from services.merkle_tree import MerkleTree
from utils.connections import get_redis

logger = get_logger(__name__)

# Default SimHash similarity threshold (Hamming distance out of 256 bits)
DEFAULT_THRESHOLD = 25

# Redis key patterns
_META_KEY = "prism:index_meta:{repo}"
_SIMHASH_SET_KEY = "prism:simhashes:{installation_id}"


class SimHashIndex:
    """Manages SimHash fingerprints for all indexed repos.

    Storage:
      - ``prism:index_meta:{repo_name}`` → JSON with simhash, merkle_root,
        installation_id, indexed_at, file_count
      - ``prism:simhashes:{installation_id}`` → Redis hash mapping repo_name
        to its simhash hex string (for fast similarity scan)
    """

    def __init__(self):
        pass

    async def _ensure_redis(self) -> Any:
        redis = get_redis()
        if redis is None:
            logger.warning("simhash_redis_pool_not_initialized")
        return redis

    async def store(
        self,
        repo_name: str,
        merkle_tree: MerkleTree,
        installation_id: int,
    ) -> None:
        """Compute and store SimHash + metadata for a repo's Merkle tree."""
        redis = await self._ensure_redis()
        if not redis:
            return

        simhash = merkle_tree.simhash()
        meta = {
            "simhash": simhash,
            "merkle_root": merkle_tree.root_hash(),
            "installation_id": installation_id,
            "indexed_at": time.time(),
            "file_count": len(merkle_tree.get_all_files()),
        }

        try:
            pipe = redis.pipeline(transaction=False)
            # Store full metadata
            pipe.set(
                _META_KEY.format(repo=repo_name),
                json.dumps(meta),
            )
            # Store simhash in the per-installation set for fast scanning
            pipe.hset(
                _SIMHASH_SET_KEY.format(installation_id=installation_id),
                repo_name,
                simhash,
            )
            await pipe.execute()

            logger.info(
                "simhash_stored",
                repo=repo_name,
                simhash=simhash[:16],
                file_count=meta["file_count"],
            )
        except Exception as e:
            logger.error("simhash_store_failed", repo=repo_name, error=str(e))

    async def store_merkle_tree(self, repo_name: str, merkle_tree: MerkleTree) -> None:
        """Store serialized Merkle tree for future diffing."""
        redis = await self._ensure_redis()
        if not redis:
            return

        try:
            tree_data = json.dumps(merkle_tree.serialize())
            await redis.set(
                f"prism:merkle:{repo_name}",
                tree_data,
                ex=86400 * 30,  # 30-day TTL
            )
        except Exception as e:
            logger.warning("merkle_store_failed", repo=repo_name, error=str(e))

    async def get_stored_merkle_tree(self, repo_name: str) -> MerkleTree | None:
        """Retrieve a previously stored Merkle tree for diffing."""
        redis = await self._ensure_redis()
        if not redis:
            return None

        try:
            tree_data = await redis.get(f"prism:merkle:{repo_name}")
            if tree_data:
                return MerkleTree.deserialize(json.loads(tree_data))
        except Exception as e:
            logger.warning("merkle_retrieve_failed", repo=repo_name, error=str(e))
        return None

    async def find_similar(
        self,
        merkle_tree: MerkleTree,
        installation_id: int,
        exclude_repo: str | None = None,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> str | None:
        """Find an already-indexed repo with SimHash Hamming distance < threshold.

        Only searches within the same GitHub App installation (same org).

        Returns:
            The repo_name of the best match, or None if no match found.
        """
        redis = await self._ensure_redis()
        if not redis:
            return None

        target_simhash = merkle_tree.simhash()

        try:
            all_simhashes: dict[str, str] = await redis.hgetall(
                _SIMHASH_SET_KEY.format(installation_id=installation_id)
            )
        except Exception as e:
            logger.warning("simhash_scan_failed", error=str(e))
            return None

        best_repo: str | None = None
        best_distance = threshold + 1  # Start above threshold

        for repo_name, simhash_hex in all_simhashes.items():
            if repo_name == exclude_repo:
                continue

            distance = MerkleTree.hamming_distance(target_simhash, simhash_hex)
            if distance < best_distance:
                best_distance = distance
                best_repo = repo_name

        if best_repo:
            logger.info(
                "simhash_match_found",
                source=best_repo,
                distance=best_distance,
                threshold=threshold,
            )

        return best_repo

    async def copy_index(
        self,
        source_repo: str,
        target_repo: str,
    ) -> int:
        """Copy all Qdrant points from source repo to target repo.

        Re-tags all points with the target repo name so they're isolated
        in queries. Returns count of points copied.
        """
        from utils.qdrant_client import get_qdrant_client

        client = get_qdrant_client()
        copied = 0

        try:
            from qdrant_client.models import (
                Filter,
                FieldCondition,
                MatchValue,
                PointStruct,
            )

            import asyncio

            # Scroll through all source repo points
            offset = None
            while True:
                points, next_offset = await asyncio.to_thread(
                    client.scroll,
                    collection_name="repo_chunks",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="repo_name",
                                match=MatchValue(value=source_repo),
                            )
                        ]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                if not points:
                    break

                # Re-tag points with target repo name and new UUIDs
                new_points = []
                for point in points:
                    payload = dict(point.payload)
                    payload["repo_name"] = target_repo
                    payload["copied_from"] = source_repo
                    new_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=point.vector,
                            payload=payload,
                        )
                    )

                if new_points:
                    await asyncio.to_thread(
                        client.upsert,
                        collection_name="repo_chunks",
                        points=new_points,
                    )
                    copied += len(new_points)

                offset = next_offset
                if offset is None:
                    break

            logger.info(
                "simhash_index_copied",
                source=source_repo,
                target=target_repo,
                points_copied=copied,
            )
        except Exception as e:
            logger.error("simhash_copy_failed", source=source_repo, error=str(e))

        return copied

    async def check_index_status(self, repo_name: str) -> str:
        """Check if a repo index exists and how fresh it is.

        Returns:
            'fresh' — indexed within the last 24 hours
            'stale' — indexed but older than 24 hours
            'none'  — never indexed
        """
        redis = await self._ensure_redis()
        if not redis:
            return "none"

        try:
            meta_raw = await redis.get(_META_KEY.format(repo=repo_name))
            if not meta_raw:
                return "none"

            meta = json.loads(meta_raw)
            age_hours = (time.time() - meta.get("indexed_at", 0)) / 3600

            if age_hours < 24:
                return "fresh"
            return "stale"

        except Exception as e:
            logger.warning("index_status_check_failed", repo=repo_name, error=str(e))
            return "none"

    async def delete_index_meta(self, repo_name: str, installation_id: int) -> None:
        """Remove a repo's index metadata (called during TTL cleanup)."""
        redis = await self._ensure_redis()
        if not redis:
            return
        try:
            pipe = redis.pipeline(transaction=False)
            pipe.delete(_META_KEY.format(repo=repo_name))
            pipe.delete(f"prism:merkle:{repo_name}")
            pipe.hdel(
                _SIMHASH_SET_KEY.format(installation_id=installation_id),
                repo_name,
            )
            await pipe.execute()
        except Exception as e:
            logger.warning("index_meta_delete_failed", repo=repo_name, error=str(e))

    async def close(self) -> None:
        # No-op: connection pool is managed centrally by utils.connections
        pass
