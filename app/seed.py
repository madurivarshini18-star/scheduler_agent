"""
Seed module — creates the Pinecone index if needed and purges any
previously inserted sample (seed-*) vectors so the app always starts
with a clean schedule. No sample events are ever inserted.
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
    """Ensure index exists and remove all old seed-* vectors."""
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
        return  # fresh index — nothing to clean

    index = pc.Index(PINECONE_INDEX_NAME)

    # Delete old sample vectors whose IDs start with "seed-"
    try:
        results = index.list(prefix="seed-", limit=200)
        ids = list(results)  # list() yields IDs directly
        if ids:
            index.delete(ids=ids)
            print(f"Removed {len(ids)} old seed vectors.")
        else:
            print("No seed vectors found — index is clean.")
    except Exception as e:
        # Older Pinecone SDK versions use a different list API — try fallback
        try:
            from app.tools import _embed
            result = index.query(
                vector=_embed("schedule events standup meeting"),
                top_k=200,
                include_metadata=True,
            )
            matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
            seed_ids = [
                (m.id if hasattr(m, "id") else m.get("id", ""))
                for m in matches
                if (m.id if hasattr(m, "id") else m.get("id", "")).startswith("seed-")
            ]
            if seed_ids:
                index.delete(ids=seed_ids)
                print(f"Removed {len(seed_ids)} old seed vectors (fallback).")
            else:
                print("No seed vectors found (fallback) — index is clean.")
        except Exception as e2:
            print(f"Seed cleanup skipped: {e2}")


if __name__ == "__main__":
    seed()
