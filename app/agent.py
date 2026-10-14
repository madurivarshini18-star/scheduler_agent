"""
agent.py — Groq-compatible scheduling agent.

Uses the Groq REST API directly with tool_choice="none" so the model
outputs plain-text JSON that we parse ourselves. This works reliably
across all current Groq free-tier models (gpt-oss-20b, gpt-oss-120b, etc.)
"""

import json
import os
import re
import urllib.request
import urllib.error
from datetime import datetime

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.tools import get_schedule, update_schedule

TOOLS = {
    "get_schedule":    get_schedule,
    "update_schedule": update_schedule,
}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are a scheduling assistant. Today is {today}.

You have two tools. Respond with ONLY a raw JSON object (no markdown, no extra text) when you need a tool:

To retrieve/query schedule:
{{"tool": "get_schedule", "args": {{"query": "2026-10-15"}}}}

To add an event:
{{"tool": "update_schedule", "args": {{"action": "add", "title": "Doctor visit", "date": "2026-10-15", "time": "10:00", "event_type": "health", "description": "fever"}}}}

To update an event:
{{"tool": "update_schedule", "args": {{"action": "update", "title": "Doctor visit", "date": "2026-10-15", "time": "11:00", "event_type": "health", "description": ""}}}}

To delete an event:
{{"tool": "update_schedule", "args": {{"action": "delete", "title": "Doctor visit", "date": "2026-10-15", "time": "", "event_type": "health", "description": ""}}}}

STRICT RULES:
1. Resolve relative dates (tomorrow, next Monday, this Friday) to YYYY-MM-DD using today = {today}.
2. When a tool is needed, output ONLY the raw JSON object. Nothing else before or after.
3. After receiving a tool result, respond naturally to the user.
4. Never invent schedule data — only use what the tool returns.
5. For greetings or questions needing no tool, answer directly."""


def _call_groq(messages: list) -> str:
    """Call Groq REST API directly with tool_choice=none (plain text output)."""
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1024,
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"] or ""


def _extract_tool_call(text: str) -> dict | None:
    """
    Extract a tool call JSON from model response text.
    Handles all known formats.
    """
    if not text or not text.strip():
        return None

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # Try the full cleaned text first
    for candidate in [cleaned, text.strip()]:
        try:
            data = json.loads(candidate)
            result = _parse_tool_dict(data)
            if result:
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to extract any {...} block from the text
    for match in re.finditer(r'\{[\s\S]*?\}', cleaned):
        try:
            data = json.loads(match.group())
            result = _parse_tool_dict(data)
            if result:
                return result
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _parse_tool_dict(data: dict) -> dict | None:
    """Convert any known tool-call dict format to {'tool': name, 'args': {...}}."""
    if not isinstance(data, dict):
        return None

    # Format 1: {"tool": "update_schedule", "args": {...}}
    if "tool" in data:
        raw_name = str(data["tool"]).replace("tool.", "").strip()
        if raw_name in TOOLS:
            args = data.get("args", {})
            # Unwrap double-nested args: {"args": {"action": ...}}
            if isinstance(args, dict) and "args" in args and isinstance(args["args"], dict):
                args = args["args"]
            return {"tool": raw_name, "args": args}

    # Format 2: {"name": "update_schedule", "arguments": {...}}
    if "name" in data:
        raw_name = str(data["name"]).replace("tool.", "").strip()
        if raw_name in TOOLS:
            args = data.get("arguments", data.get("args", {}))
            if isinstance(args, dict) and "args" in args and isinstance(args["args"], dict):
                args = args["args"]
            return {"tool": raw_name, "args": args}

    return None


def _run_tool(tool_call: dict) -> str:
    """Execute a tool and return result as string."""
    name    = tool_call.get("tool", "")
    args    = tool_call.get("args", {})
    tool_fn = TOOLS.get(name)
    if not tool_fn:
        return f"Unknown tool: {name}"
    try:
        return str(tool_fn.invoke(args))
    except Exception as e:
        return f"Tool error: {e}"


def run_agent(user_input: str) -> str:
    """Main agent loop — up to 5 iterations."""
    messages = [
        {"role": "system",  "content": _system_prompt()},
        {"role": "user",    "content": user_input},
    ]

    for step in range(5):
        try:
            reply = _call_groq(messages)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            # Groq sometimes returns 400 with the tool call in failed_generation
            tool_call = _extract_tool_call_from_groq_error(body)
            if tool_call:
                result = _run_tool(tool_call)
                messages.append({"role": "assistant", "content": json.dumps(tool_call)})
                messages.append({"role": "user",      "content": f"Tool result for {tool_call['tool']}:\n{result}"})
                continue
            return f"API error ({e.code}): {body[:300]}"
        except Exception as e:
            return f"Agent error: {e}"

        messages.append({"role": "assistant", "content": reply})

        tool_call = _extract_tool_call(reply)

        if not tool_call:
            # No tool needed — final answer
            return reply.strip() or "Done."

        # Execute the tool
        result = _run_tool(tool_call)
        messages.append({"role": "user", "content": f"Tool result for {tool_call['tool']}:\n{result}"})

    return "Reached maximum steps. Please try a simpler request."


def _extract_tool_call_from_groq_error(error_body: str) -> dict | None:
    """Extract tool call from Groq 400 error 'failed_generation' field."""
    try:
        err_data = json.loads(error_body)
        failed   = err_data.get("error", {}).get("failed_generation", "")
        if failed:
            return _extract_tool_call(failed)
    except Exception:
        pass
    # Fallback: search raw string
    match = re.search(r'failed_generation["\s:]+(\{.*?\})', error_body, re.DOTALL)
    if match:
        return _extract_tool_call(match.group(1))
    return None
