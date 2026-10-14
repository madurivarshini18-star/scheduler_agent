import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


GROQ_API_KEY = _require("GROQ_API_KEY").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

PINECONE_API_KEY = _require("PINECONE_API_KEY").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "schedule-agent").strip()
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip()
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1").strip()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
EMBEDDING_DIMENSION = 384
