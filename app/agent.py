import json
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.tools import get_schedule, update_schedule

TOOLS = {
    "get_schedule": get_schedule,
    "update_schedule": update_schedule,
}

# Bind tools natively so any Groq model can call them correctly
_base_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
try:
    llm = _base_llm.bind_tools([get_schedule, update_schedule])
except Exception:
    llm = _base_llm


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are a helpful scheduling assistant. Today is {today}.

Use the available tools to manage the user's schedule:
- Use get_schedule to retrieve or query events
- Use update_schedule to add, update, or delete events

Rules:
1. Always resolve relative dates (tomorrow, next Monday, etc.) to exact YYYY-MM-DD using today ({today}).
2. For any query about schedule or availability, call get_schedule.
3. For adding/updating/deleting events, call update_schedule.
4. After receiving tool results, answer the user naturally.
5. Never invent schedule data — only use what the tool returns.
6. If no tool is needed, answer directly."""


def _extract_native_tool_calls(response) -> list:
    """Extract tool calls from native LangChain tool_calls attribute."""
    calls = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            calls.append({
                "id":   tc.get("id", "tc_0"),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            })
    return calls


def _extract_json_tool_call(text: str) -> dict | None:
    """Fallback: extract plain JSON tool call from text (older approach)."""
    # Try nested JSON: {"tool": "...", "args": {...}}
    match = re.search(r'\{[^{}]*"tool"\s*:[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass
    # Try top-level JSON with "name" key (native format in text)
    try:
        data = json.loads(text.strip())
        if "name" in data and data["name"] in TOOLS:
            return {"tool": data["name"], "args": data.get("arguments", data.get("args", {}))}
        if "tool" in data:
            return data
    except json.JSONDecodeError:
        pass
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
            return f"Agent error: {str(e)}"

        content = response.content if isinstance(response.content, str) else str(response.content)

        # ── Try native tool calls first (works with gpt-oss, newer models) ──
        native_calls = _extract_native_tool_calls(response)

        if native_calls:
            messages.append(response)  # append AIMessage with tool_calls
            all_results = []

            for tc in native_calls:
                name    = tc["name"]
                args    = tc["args"]
                tool_id = tc["id"]
                tool_fn = TOOLS.get(name)

                if tool_fn:
                    try:
                        result = tool_fn.invoke(args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                else:
                    result = f"Unknown tool: {name}"

                all_results.append(result)
                # Add ToolMessage for each tool call (required by LangChain)
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

            continue  # Let LLM generate final answer using tool results

        # ── Fallback: plain JSON in text (older approach) ──
        json_call = _extract_json_tool_call(content)

        if json_call:
            messages.append(AIMessage(content=content))
            name    = json_call.get("tool", "")
            args    = json_call.get("args", {})
            tool_fn = TOOLS.get(name)

            if tool_fn:
                try:
                    result = tool_fn.invoke(args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {name}"

            messages.append(HumanMessage(content=f"Tool result for {name}:\n{result}"))
            continue

        # ── No tool call — this is the final answer ──
        return content.strip() or "I couldn't process that request."

    return "I reached the maximum steps. Please try a more specific question."
