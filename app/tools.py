"""
Two agent tools:
- get_schedule(query)  : Retrieve schedule by date or natural language query.
- update_schedule(...) : Add, update, or remove schedule entries.
"""

import uuid
from datetime import datetime

from fastembed import TextEmbedding
from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec

from app.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)
_embedder = TextEmbedding(model_name=EMBEDDING_MODEL)


def _ensure_index():
    existing = [idx.name for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return _pc.Index(PINECONE_INDEX_NAME)


_index = _ensure_index()


def _embed(text: str) -> list[float]:
    return list(_embedder.embed([text]))[0].tolist()


def _normalize_date(date_str: str) -> str:
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
                "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _format_matches(matches) -> str:
    if not matches:
        return ""
    lines = []
    seen = set()
    for m in matches:
        meta = m.metadata if hasattr(m, "metadata") else m.get("metadata", {})
        mid = m.id if hasattr(m, "id") else m.get("id", "")
        if mid in seen:
            continue
        seen.add(mid)
        date = meta.get("date", "?")
        time = meta.get("time", "?")
        title = meta.get("title", "?")
        etype = meta.get("event_type", "event")
        desc = meta.get("description", "")
        line = f"[{etype.upper()}] {date} {time} - {title}"
        if desc:
            line += f" | {desc}"
        lines.append((date, time, line))
    lines.sort(key=lambda x: (x[0], x[1]))
    return "\n".join(l[2] for l in lines)


@tool
def get_schedule(query: str) -> str:
    """Retrieve schedule entries by date or natural language query.

    Use for questions like "What do I have tomorrow?", "Am I free Friday afternoon?",
    "Show my meetings next week", "Do I have anything on 2026-08-20?".

    Args:
        query: A date (YYYY-MM-DD) or natural language like "tomorrow morning".

    Returns:
        Formatted list of matching schedule entries, or a free message.
    """
    norm = _normalize_date(query)
    is_date = len(norm) == 10 and norm[4] == "-" and norm[7] == "-"

    if is_date:
        result = _index.query(
            vector=_embed(query),
            top_k=20,
            filter={"date": {"$eq": norm}},
            include_metadata=True,
        )
    else:
        result = _index.query(
            vector=_embed(query),
            top_k=10,
            include_metadata=True,
        )

    matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
    formatted = _format_matches(matches)
    if not formatted:
        label = norm if is_date else f"'{query}'"
        return f"No schedule entries found for {label}. You look free!"
    return formatted


@tool
def update_schedule(
    action: str,
    title: str,
    date: str,
    time: str = "09:00",
    event_type: str = "meeting",
    description: str = "",
    entry_id: str = "",
) -> str:
    """Add, update, or remove a schedule entry.

    Args:
        action: "add", "update", or "delete".
        title: Event title.
        date: Date in YYYY-MM-DD format.
        time: Time in HH:MM format. Default "09:00".
        event_type: Category e.g. "meeting", "appointment", "task", "workshop".
        description: Optional extra detail.
        entry_id: For update/delete, ID of existing entry (auto-resolved if omitted).

    Returns:
        Confirmation of the action taken.
    """
    norm_date = _normalize_date(date)
    action = action.lower().strip()

    if action == "add":
        new_id = f"{norm_date}-{uuid.uuid4().hex[:8]}"
        text = f"{event_type} on {norm_date} at {time}: {title}. {description}"
        _index.upsert(vectors=[{
            "id": new_id,
            "values": _embed(text),
            "metadata": {
                "date": norm_date,
                "time": time,
                "title": title,
                "event_type": event_type,
                "description": description,
            },
        }])
        return f"Added [{event_type.upper()}] '{title}' on {norm_date} at {time}. (ID: {new_id})"

    elif action == "update":
        if not entry_id:
            result = _index.query(
                vector=_embed(f"{title} {norm_date}"),
                top_k=5,
                filter={"date": {"$eq": norm_date}},
                include_metadata=True,
            )
            matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
            if not matches:
                return f"Could not find '{title}' on {norm_date} to update."
            entry_id = matches[0].id if hasattr(matches[0], "id") else matches[0].get("id")
        _index.delete(ids=[entry_id])
        new_id = f"{norm_date}-{uuid.uuid4().hex[:8]}"
        text = f"{event_type} on {norm_date} at {time}: {title}. {description}"
        _index.upsert(vectors=[{
            "id": new_id,
            "values": _embed(text),
            "metadata": {
                "date": norm_date,
                "time": time,
                "title": title,
                "event_type": event_type,
                "description": description,
            },
        }])
        return f"Updated '{title}' on {norm_date} - now at {time}. (New ID: {new_id})"

    elif action == "delete":
        if not entry_id:
            result = _index.query(
                vector=_embed(f"{title} {norm_date}"),
                top_k=5,
                filter={"date": {"$eq": norm_date}},
                include_metadata=True,
            )
            matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
            if not matches:
                return f"Could not find '{title}' on {norm_date} to delete."
            entry_id = matches[0].id if hasattr(matches[0], "id") else matches[0].get("id")
        _index.delete(ids=[entry_id])
        return f"Deleted '{title}' on {norm_date}. (ID: {entry_id})"

    else:
        return f"Unknown action '{action}'. Use 'add', 'update', or 'delete'."
