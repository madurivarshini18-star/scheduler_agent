"""
Seed disabled — sample events removed.
"""

from pinecone import Pinecone, ServerlessSpec
from app.config import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)


def seed():
    """Create the Pinecone index if it doesn't exist. Does NOT insert any sample data."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists — no seeding.")


if __name__ == "__main__":
    seed()
