"""
embeddings.py — Shared embedding utility

Uses ChromaDB's built-in ONNX-based embedding function (all-MiniLM-L6-v2)
instead of sentence-transformers + PyTorch. This avoids the ~2GB torch
dependency while using the exact same underlying model.

Includes a robust pre-downloader for the ONNX model that handles slow
connections better than Chroma's built-in httpx-based downloader.
"""

import os
import sys
import tarfile
import urllib.request
import numpy as np

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# Model download constants (must match Chroma's expectations)
_MODEL_NAME = "all-MiniLM-L6-v2"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "chroma", "onnx_models", _MODEL_NAME)
_ARCHIVE_PATH = os.path.join(_CACHE_DIR, "onnx.tar.gz")
_EXTRACTED_DIR = os.path.join(_CACHE_DIR, "onnx")
_MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"

# Lazy-loaded singleton
_embed_fn = None


def _ensure_onnx_model():
    """Pre-download the ONNX model if not already cached.

    Uses urllib with resume support and retry logic for unstable connections.
    The download can survive connection resets — it picks up from where
    it left off using HTTP Range headers.
    """
    import time

    # Check if model is already extracted
    if os.path.exists(_EXTRACTED_DIR):
        return

    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp_path = _ARCHIVE_PATH + ".tmp"

    # Download the archive if not present
    if not os.path.exists(_ARCHIVE_PATH):
        print(f"Downloading ONNX embedding model ({_MODEL_NAME})...")
        print(f"This is a one-time download (~79MB). Please wait...")

        max_retries = 10
        retry_delay = 5  # seconds

        for attempt in range(1, max_retries + 1):
            # Check how much we already downloaded (for resume)
            downloaded = 0
            if os.path.exists(tmp_path):
                downloaded = os.path.getsize(tmp_path)

            headers = {"User-Agent": "arxiv-digest-agent/1.0"}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"  Resuming from {downloaded / (1024*1024):.1f} MB (attempt {attempt}/{max_retries})...")

            req = urllib.request.Request(_MODEL_URL, headers=headers)

            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    # Get total size from Content-Range or Content-Length
                    content_range = response.headers.get("Content-Range", "")
                    if content_range:
                        total_size = int(content_range.split("/")[-1])
                    else:
                        total_size = int(response.headers.get("Content-Length", 0)) + downloaded

                    chunk_size = 65536  # 64KB chunks
                    mode = "ab" if downloaded > 0 else "wb"

                    with open(tmp_path, mode) as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = downloaded * 100 // total_size
                                mb_done = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                sys.stdout.write(
                                    f"\r  Progress: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)"
                                )
                                sys.stdout.flush()

                # If we got here, download completed
                os.replace(tmp_path, _ARCHIVE_PATH)
                print("\n  Download complete.")
                break  # Exit retry loop

            except Exception as e:
                print(f"\n  Connection interrupted: {e}")
                if attempt < max_retries:
                    print(f"  Retrying in {retry_delay}s (attempt {attempt}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to download ONNX model after {max_retries} attempts: {e}\n"
                        f"You can manually download it from:\n  {_MODEL_URL}\n"
                        f"And extract to:\n  {_CACHE_DIR}"
                    ) from e

    # Extract the archive
    print("  Extracting model...")
    try:
        with tarfile.open(_ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=_CACHE_DIR)
        print("  Model ready.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract ONNX model archive: {e}") from e


def _get_embed_fn() -> ONNXMiniLM_L6_V2:
    """Lazy-load the ONNX embedding function (avoids loading at import time)."""
    global _embed_fn
    if _embed_fn is None:
        _ensure_onnx_model()
        _embed_fn = ONNXMiniLM_L6_V2()
    return _embed_fn


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning a numpy array of shape (n, dim).

    Embeddings are L2-normalized for cosine similarity.
    """
    ef = _get_embed_fn()
    embeddings = ef(texts)
    arr = np.array(embeddings, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return arr / norms


def embed_query(text: str) -> np.ndarray:
    """Embed a single query text, returning a 1D numpy array."""
    return embed_texts([text])[0]


def get_chroma_embedding_function() -> ONNXMiniLM_L6_V2:
    """Return the embedding function for use with Chroma collections.

    Pass this to `client.create_collection(embedding_function=...)` so
    Chroma uses the same model for automatic embedding on add/query.
    """
    return _get_embed_fn()
