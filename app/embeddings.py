"""
app/embeddings.py
-----------------
Handles embedding generation and semantic search over the SHL product catalog.

Pipeline:
  1. load_catalog()  → list of dicts from JSON
  2. build_index()   → encode catalog texts → FAISS index + metadata.json (saved to data/)
  3. load_index()    → restore index + metadata from disk
  4. search()        → embed a query and return top-k catalog items

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dim embeddings, fast on CPU, good semantic similarity for short texts
  - Inner-product search after L2 normalization ≡ cosine similarity
"""

import json
import os
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (everything lives under the project root's  data/  directory)
# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the module works regardless of CWD.
_HERE = Path(__file__).resolve().parent          # …/app/
_DATA_DIR = _HERE.parent / "data"                # …/data/
CATALOG_PATH = _DATA_DIR / "shl_catalog.json"
FAISS_INDEX_PATH = _DATA_DIR / "faiss.index"
METADATA_PATH = _DATA_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Model name
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Key → single-letter test_type mapping  (based on SHL catalog conventions)
# ---------------------------------------------------------------------------
KEY_TO_TYPE: dict[str, str] = {
    "Ability & Aptitude":           "A",
    "Assessment Exercises":         "E",
    "Biodata & Situational Judgment": "B",
    "Competencies":                 "C",
    "Development & 360":            "D",
    "Knowledge & Skills":           "K",
    "Personality & Behavior":       "P",
    "Simulations":                  "S",
}


# ===========================================================================
# 1. Catalog loader
# ===========================================================================

def load_catalog(path: str | Path = CATALOG_PATH) -> list[dict[str, Any]]:
    """
    Load the SHL product catalog from a JSON file.

    Parameters
    ----------
    path : str or Path
        Location of shl_catalog.json.  Defaults to data/shl_catalog.json.

    Returns
    -------
    list[dict]
        Raw catalog records.  Each record is expected to have at minimum:
        entity_id, name, link, description, keys, job_levels.

    Raises
    ------
    FileNotFoundError  – if the file does not exist.
    ValueError         – if the file is not valid JSON or is not a list.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            catalog = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in catalog file: {exc}") from exc

    if not isinstance(catalog, list):
        raise ValueError("Catalog JSON must be a top-level array of objects.")

    logger.info("Loaded %d items from catalog (%s)", len(catalog), path)
    return catalog


# ===========================================================================
# 2. Text builder
# ===========================================================================

def _build_searchable_text(item: dict[str, Any]) -> str:
    """
    Combine the most semantically meaningful fields of a catalog item into a
    single string that the embedding model will encode.

    Fields used (in order of descending importance):
      - name         : most discriminative signal; repeated for weight
      - description  : free-text summary of what the assessment measures
      - keys         : test-type categories (e.g. "Personality & Behavior")
      - job_levels   : target seniority levels (e.g. "Manager, Director")

    The sentence structure ("… measures …", "… suitable for …") helps the
    model align the catalog text with natural-language hiring queries.
    """
    name = item.get("name", "").strip()
    description = item.get("description", "").strip()

    # keys is a list like ["Personality & Behavior", "Knowledge & Skills"]
    keys_list = item.get("keys", [])
    keys_str = ", ".join(keys_list) if keys_list else "Unknown"

    # job_levels is a list like ["Manager", "Director"]
    levels_list = item.get("job_levels", [])
    levels_str = ", ".join(levels_list) if levels_list else "All levels"

    # Structured sentence that mirrors how a hiring manager might phrase a query
    text = (
        f"{name}. "
        f"{description} "
        f"Test type: {keys_str}. "
        f"Suitable for: {levels_str}."
    )
    return text


def _derive_test_type(item: dict[str, Any]) -> str:
    """
    Convert the keys list to a slash-separated string of single-letter codes.
    E.g. ["Personality & Behavior", "Knowledge & Skills"] → "P/K"
    Falls back to "?" if no mapping exists.
    """
    codes = [KEY_TO_TYPE.get(k, "?") for k in item.get("keys", [])]
    return "/".join(codes) if codes else "?"


# ===========================================================================
# 3. Build index (run once / on catalog update)
# ===========================================================================

def build_index(
    catalog_path: str | Path = CATALOG_PATH,
    index_path: str | Path = FAISS_INDEX_PATH,
    metadata_path: str | Path = METADATA_PATH,
    model_name: str = EMBEDDING_MODEL,
) -> None:
    """
    Build a FAISS index over the SHL catalog and persist it to disk.

    Steps:
      1. Load catalog via load_catalog().
      2. Build a searchable text string for every item.
      3. Encode all texts with the sentence-transformer model.
      4. L2-normalize the vectors so inner-product search = cosine similarity.
      5. Create a flat FAISS index (IndexFlatIP) and add the vectors.
      6. Save the index to  data/faiss.index.
      7. Save lightweight metadata (no embeddings) to  data/metadata.json.

    Parameters
    ----------
    catalog_path   : path to shl_catalog.json
    index_path     : destination for faiss.index
    metadata_path  : destination for metadata.json
    model_name     : HuggingFace model identifier

    Notes
    -----
    - IndexFlatIP with L2-normalized vectors gives exact cosine similarity.
      With only 377 items an approximate index (IVF/HNSW) would be overkill.
    - The metadata list preserves insertion order, so metadata[i] corresponds
      to the vector at row i of the FAISS index.
    """
    catalog = load_catalog(catalog_path)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build searchable texts + lightweight metadata records
    # ------------------------------------------------------------------
    texts: list[str] = []
    metadata: list[dict[str, Any]] = []

    for item in catalog:
        texts.append(_build_searchable_text(item))

        metadata.append({
            "entity_id":  item.get("entity_id", ""),
            "name":        item.get("name", ""),
            "url":         item.get("link", ""),
            "description": item.get("description", ""),
            "keys":        item.get("keys", []),
            "test_type":   _derive_test_type(item),
            "job_levels":  item.get("job_levels", []),
            "duration":    item.get("duration", ""),
            "remote":      item.get("remote", ""),
            "adaptive":    item.get("adaptive", ""),
            "languages":   item.get("languages", []),
        })

    logger.info("Built searchable texts for %d catalog items.", len(texts))

    # ------------------------------------------------------------------
    # Load model and encode
    # ------------------------------------------------------------------
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Encoding %d texts …", len(texts))
    # show_progress_bar=True gives a tqdm bar in the terminal
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-norm in-place → cosine via IP
    )
    logger.info("Embeddings shape: %s  dtype: %s", embeddings.shape, embeddings.dtype)

    # ------------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------------
    dim = embeddings.shape[1]                    # 384 for MiniLM-L6-v2
    index = faiss.IndexFlatIP(dim)               # exact inner-product (= cosine after norm)
    index.add(embeddings.astype(np.float32))     # FAISS requires float32
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

    # ------------------------------------------------------------------
    # Persist to disk
    # ------------------------------------------------------------------
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved → %s", index_path)

    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    logger.info("Metadata saved → %s  (%d records)", metadata_path, len(metadata))


# ===========================================================================
# 4. Load index from disk
# ===========================================================================

def load_index(
    index_path: str | Path = FAISS_INDEX_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """
    Load a previously built FAISS index and its associated metadata from disk.

    Parameters
    ----------
    index_path    : path to faiss.index
    metadata_path : path to metadata.json

    Returns
    -------
    (faiss.Index, list[dict])
        The FAISS index object and the list of metadata dicts (same order as
        vectors in the index).

    Raises
    ------
    FileNotFoundError – if either file is missing (call build_index() first).
    """
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run build_index() first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. Run build_index() first."
        )

    index = faiss.read_index(str(index_path))
    logger.info("FAISS index loaded: %d vectors, dim=%d", index.ntotal, index.d)

    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata: list[dict[str, Any]] = json.load(fh)
    logger.info("Metadata loaded: %d records", len(metadata))

    return index, metadata


# ===========================================================================
# 5. Search
# ===========================================================================

# Module-level singletons — populated lazily on first search() call so the
# FastAPI app doesn't pay the load cost at import time.
_index: faiss.Index | None = None
_metadata: list[dict[str, Any]] | None = None
_model: SentenceTransformer | None = None


def _ensure_loaded() -> None:
    """
    Lazy-load the FAISS index, metadata, and embedding model into module-level
    singletons.  Subsequent calls are no-ops (already loaded check).
    """
    global _index, _metadata, _model

    if _index is None or _metadata is None:
        _index, _metadata = load_index()

    if _model is None:
        logger.info("Loading embedding model for search: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)


def search(
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Perform a semantic similarity search over the SHL catalog.

    The query is embedded with the same model used at index-build time.
    FAISS returns the top-k most similar catalog items by cosine similarity
    (inner product of L2-normalized vectors).

    Parameters
    ----------
    query  : natural-language hiring query, e.g.
             "I need a personality test for a mid-level sales manager"
    top_k  : maximum number of results to return (default: 10, max: 10 per spec)

    Returns
    -------
    list[dict]
        Ordered list of metadata dicts from closest to least similar.
        Each dict includes: entity_id, name, url, description, keys,
        test_type, job_levels, duration, remote, adaptive, languages.

    Raises
    ------
    ValueError  – if query is empty or top_k is not a positive integer.
    RuntimeError – if the index has not been built yet.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # Clamp to catalog size to avoid FAISS assertion errors on tiny catalogs
    try:
        _ensure_loaded()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Search index not available. Call build_index() to create it."
        ) from exc

    actual_k = min(top_k, _index.ntotal)    # type: ignore[union-attr]

    # Encode query — must match index normalization (normalize_embeddings=True)
    query_vec: np.ndarray = _model.encode(  # type: ignore[union-attr]
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)                     # shape: (1, 384)

    # FAISS search returns (distances, indices) as shape (1, k) arrays
    distances, indices = _index.search(query_vec, actual_k)  # type: ignore[union-attr]

    results: list[dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 for unfilled slots (shouldn't happen with FlatIP)
            continue
        record = dict(_metadata[idx])       # type: ignore[index]
        record["score"] = float(dist)       # cosine similarity ∈ [-1, 1]
        results.append(record)

    logger.debug("search('%s', top_k=%d) → %d results", query[:60], top_k, len(results))
    return results


# ===========================================================================
# CLI entry-point: python -m app.embeddings  (or  python app/embeddings.py)
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=== Building SHL FAISS index ===")
    build_index()
    print("\n=== Quick smoke-test ===")

    test_queries = [
        "Java developer with stakeholder management skills",
        "personality test for senior sales manager",
        "cognitive ability test for entry-level graduate",
        "numerical reasoning for finance analyst",
    ]

    for q in test_queries:
        hits = search(q, top_k=3)
        print(f"\nQuery : {q}")
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['name']}  |  {h['test_type']}  |  {h['url']}")

    sys.exit(0)
