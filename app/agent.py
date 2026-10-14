import json
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.tools import get_schedule, update_schedule

TOOLS = {
    "get_schedule": get_schedule,
    "update_schedule": update_schedule,
}

# Plain LLM — NO bind_tools, no tool schemas sent to the model
# This avoids the 'tool_use_failed' error with gpt-oss models
llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are a scheduling assistant. Today is {today}.

You have two tools. When you need a tool, respond with ONLY a raw JSON object — no markdown, no explanation, nothing else:

To retrieve schedule:
{{"tool": "get_schedule", "args": {{"query": "2026-10-15"}}}}

To add/update/delete an event:
{{"tool": "update_schedule", "args": {{"action": "add", "title": "Doctor visit", "date": "2026-10-15", "time": "10:00", "event_type": "health", "description": "fever"}}}}

STRICT RULES:
1. Resolve all relative dates to YYYY-MM-DD format using today = {today}.
2. Output ONLY the raw JSON object when calling a tool. No other text.
3. After receiving a tool result, give a natural language answer.
4. Never invent data. Only use what the tool returns.
5. If no tool is needed, answer directly in plain text."""


def _extract_tool_call(text: str) -> dict | None:
    """
    Extract tool call from model response.
    Handles multiple formats:
    - {"tool": "update_schedule", "args": {...}}
    - {"name": "update_schedule", "arguments": {...}}
    - {"name": "tool.update_schedule", "arguments": {...}}
    """
    if not text or not text.strip():
        return None

    # Try to find a JSON block anywhere in the text
    # First try to extract any {...} block
    candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)

    # Also try the full text as JSON
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

        # Format 1: {"tool": "update_schedule", "args": {...}}
        if "tool" in data and data["tool"] in TOOLS:
            return {"tool": data["tool"], "args": data.get("args", {})}

        # Format 2: {"name": "update_schedule", "arguments": {...}}
        # Format 3: {"name": "tool.update_schedule", "arguments": {...}}
        if "name" in data:
            # Strip "tool." prefix if present
            raw_name = data["name"].replace("tool.", "")
            if raw_name in TOOLS:
                args = data.get("arguments", data.get("args", {}))
                # Handle nested args: {"tool": "x", "args": {...}}
                if isinstance(args, dict) and "tool" in args and "args" in args:
                    args = args["args"]
                return {"tool": raw_name, "args": args}

        # Format 4: nested inside failed_generation style
        # {"tool": "update_schedule", "args": {"action": ..., "args": {...}}}
        if "tool" in data:
            raw_name = str(data["tool"]).replace("tool.", "")
            if raw_name in TOOLS:
                args = data.get("args", {})
                # Unwrap double-nested args if present
                if "args" in args and isinstance(args["args"], dict):
                    args = args["args"]
                return {"tool": raw_name, "args": args}

    return None


def run_agent(user_input: str) -> str:
    messages = [
        SystemMessage(content=_system_prompt()),
        HumanMessage(content=user_input),
    ]

    for _ in range(5):
        try:
            response = llm.invoke(messages)
        except Exception as e:
            err = str(e)
            # Model tried to call a tool natively — extract from error message
            tool_call = _extract_tool_call_from_error(err)
            if tool_call:
                result = _run_tool(tool_call)
                messages.append(HumanMessage(content=f"Tool result for {tool_call['tool']}:\n{result}"))
                continue
            return f"Agent error: {err}"

        content = response.content if isinstance(response.content, str) else str(response.content)
        messages.append(AIMessage(content=content))

        tool_call = _extract_tool_call(content)
        if not tool_call:
            return content.strip() or "Done."

        result = _run_tool(tool_call)
        messages.append(HumanMessage(content=f"Tool result for {tool_call['tool']}:\n{result}"))

    return "I reached the maximum steps. Please try again."


def _run_tool(tool_call: dict) -> str:
    """Execute a tool call and return the result string."""
    name    = tool_call.get("tool", "")
    args    = tool_call.get("args", {})
    tool_fn = TOOLS.get(name)
    if tool_fn:
        try:
            return tool_fn.invoke(args)
        except Exception as e:
            return f"Tool error: {e}"
    return f"Unknown tool: {name}"


def _extract_tool_call_from_error(error_str: str) -> dict | None:
    """
    When the model sends a native tool call that Groq rejects,
    the error message contains the JSON in 'failed_generation'.
    Parse it from there.
    """
    # Look for JSON in the error string
    match = re.search(r"'failed_generation':\s*'(\{.*?\})'", error_str, re.DOTALL)
    if not match:
        match = re.search(r'"failed_generation":\s*"(\{.*?\})"', error_str, re.DOTALL)
    if match:
        try:
            raw = match.group(1).replace("\\'", "'")
            return _extract_tool_call(raw)
        except Exception:
            pass

    # Try extracting any JSON from the error string
    return _extract_tool_call(error_str)
