"""Chunk articles and load them into a local Weaviate vector index.

Weaviate runs "embedded" (a local subprocess Weaviate itself manages, no
Docker) with data persisted under data_dir/weaviate. We provide our own
fastembed vectors (`Configure.Vectors.self_provided`) rather than using one
of Weaviate's built-in vectorizer integrations, since embedding is already
handled by indexer.py/retriever.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import weaviate
from fastembed import TextEmbedding
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.exceptions import WeaviateStartUpError

from faq_rag.chunker import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS, chunk_article
from faq_rag.parser import Article

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "Chunk"
# connect_to_embedded's defaults, pinned so the fallback below uses the same
# ports (the client's EmbeddedOptions class claims a different gRPC default).
EMBEDDED_HTTP_PORT = 8079
EMBEDDED_GRPC_PORT = 50050


def connect(data_dir: Path) -> weaviate.WeaviateClient:
    """Connect to (starting, if needed) the local embedded Weaviate instance."""
    weaviate_dir = data_dir / "weaviate"
    weaviate_dir.mkdir(parents=True, exist_ok=True)
    try:
        return weaviate.connect_to_embedded(
            port=EMBEDDED_HTTP_PORT,
            grpc_port=EMBEDDED_GRPC_PORT,
            persistence_data_path=str(weaviate_dir),
            environment_variables={"LOG_LEVEL": "error"},
        )
    except WeaviateStartUpError:
        # Embedded mode runs one server per data dir, owned by whichever
        # process started it - e.g. an open `faq-rag chat`. Share that server
        # instead of failing (closing this client doesn't stop it).
        return weaviate.connect_to_local(port=EMBEDDED_HTTP_PORT, grpc_port=EMBEDDED_GRPC_PORT)


def build_index(
    articles_path: Path,
    data_dir: Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> int:
    """Chunk every article, embed the chunks, and load them into Weaviate
    (the collection is dropped and recreated each run). Returns the chunk count."""
    articles = load_articles(articles_path)
    if not articles:
        raise ValueError(f"No articles found in {articles_path} - run `faq-rag crawl` first.")

    chunks = [c for article in articles for c in chunk_article(article, max_chars, overlap_chars)]

    print(f"Embedding {len(chunks)} chunks from {len(articles)} articles with {EMBEDDING_MODEL_NAME}...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    vectors = np.array(list(model.embed([c.text for c in chunks])), dtype=np.float32)
    vectors = _normalize(vectors)

    print("Loading into Weaviate...")
    client = connect(data_dir)
    try:
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
        client.collections.create(
            COLLECTION_NAME,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
            ),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="url", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="section", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
        )
        collection = client.collections.use(COLLECTION_NAME)

        objects = [
            DataObject(
                properties={
                    "chunk_id": chunk.chunk_id,
                    "url": chunk.url,
                    "title": chunk.title,
                    "section": chunk.section,
                    "category": chunk.category,
                    "text": chunk.text,
                },
                vector=vector.tolist(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        result = collection.data.insert_many(objects)
        if result.errors:
            first_error = next(iter(result.errors.values()))
            raise RuntimeError(f"{len(result.errors)} chunks failed to load into Weaviate: {first_error.message}")
    finally:
        client.close()

    return len(chunks)


def load_articles(articles_path: Path) -> list[Article]:
    """One Article per URL. The file is append-only, so when a page was
    re-crawled (e.g. after a parser fix) the later, newer record wins."""
    if not articles_path.exists():
        return []
    by_url: dict[str, Article] = {}
    with articles_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                article = Article(**json.loads(line))
                by_url[article.url] = article
    return list(by_url.values())


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
