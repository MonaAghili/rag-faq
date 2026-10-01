"""Query the local Weaviate index for the top-k most similar chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from weaviate.classes.query import MetadataQuery

from faq_rag.indexer import COLLECTION_NAME, EMBEDDING_MODEL_NAME, connect

# bge models were trained to prepend this to queries (but not to passages) so
# the query and passage embedding spaces line up correctly.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    url: str
    title: str
    section: str
    category: str
    text: str


class Retriever:
    """Connects to the local embedded Weaviate index; each search embeds the
    query and asks Weaviate's HNSW index for the nearest chunks by cosine
    distance (converted to a 0..1 similarity score: score = 1 - distance)."""

    def __init__(self, data_dir: Path):
        self.client = connect(data_dir)
        if not self.client.collections.exists(COLLECTION_NAME):
            self.client.close()
            raise FileNotFoundError(f"No index found under {data_dir} - run `faq-rag index` first.")
        self.collection = self.client.collections.use(COLLECTION_NAME)
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Retriever:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = next(self.model.embed([QUERY_PREFIX + query]))
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)

        response = self.collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            props = obj.properties
            results.append(
                RetrievedChunk(
                    score=1.0 - distance,
                    chunk_id=props["chunk_id"],
                    url=props["url"],
                    title=props["title"],
                    section=props["section"],
                    category=props["category"],
                    text=props["text"],
                )
            )
        return results
