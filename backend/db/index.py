"""
Prism Vector Database Collection Initialization.

"""

from observability.logging import get_logger
from utils.qdrant_client import qdrant_client

logger = get_logger(__name__)

VECTOR_SIZE = 384
COLLECTIONS = {
    "code_graphs": {
        "description": "Code structure graphs with functions, classes and calls",
        "vector_size": VECTOR_SIZE,
    },
    "import_files": {
        "description": "Source code files with their import dependencies",
        "vector_size": VECTOR_SIZE,
    },
    "learnings": {
        "description": "Learnings from PR commits & comments along with user feedback",
        "vector_size": VECTOR_SIZE,
    },
}


def initialize_collections() -> None:
    """
    Create Qdrant collections if they don't exist.
    Gracefully handles Qdrant being unavailable (NoOp client).
    """
    try:
        from qdrant_client.models import (
            Distance,
            PayloadSchemaType,
            QuantizationConfig,
            ScalarQuantization,
            ScalarType,
            VectorParams,
        )
    except ImportError:
        logger.warning("qdrant_sdk_missing", msg="qdrant-client not installed, skipping collection init")
        return

    from utils.config import settings

    try:
        existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    except Exception as e:
        logger.warning("qdrant_init_skipped", error=str(e), msg="Cannot reach Qdrant, skipping collection init")
        return

    for collection_name, config in COLLECTIONS.items():
        if collection_name not in existing_collections:
            logger.info("creating_collection", collection=collection_name)
            try:
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
            except Exception as e:
                logger.error("collection_creation_failed", collection=collection_name, error=str(e))
                continue
        else:
            logger.debug("collection_exists", collection=collection_name)

        # Create payload indexes for filtering
        try:
            if collection_name in ["code_graphs", "import_files"]:
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="file_path",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.debug("payload_index_created", collection=collection_name, field="file_path")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.debug("payload_index_note", collection=collection_name, error=str(e))
