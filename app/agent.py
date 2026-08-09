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

llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)


def _system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are a scheduling assistant. Today is {today}.

You can call two tools by responding ONLY with a JSON block like this (no other text):

To retrieve schedule:
{{"tool": "get_schedule", "args": {{"query": "YYYY-MM-DD"}}}}

To add/update/delete an event:
{{"tool": "update_schedule", "args": {{"action": "add", "title": "Event title", "date": "YYYY-MM-DD", "time": "HH:MM", "event_type": "meeting", "description": ""}}}}

Rules:
1. Always resolve relative dates to exact YYYY-MM-DD using today ({today}) as reference.
2. For any query about schedule or availability, output ONLY the get_schedule JSON.
3. For adding/updating/deleting events, output ONLY the update_schedule JSON.
4. After receiving tool results, answer the user naturally based on those results.
5. Never invent schedule data — only use what the tool returns.
6. If no tool is needed, answer directly."""


def _extract_tool_call(text: str) -> dict | None:
    match = re.search(r'\{{[^{{}}]*"tool"\s*:[^{{}}]*\}}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        data = json.loads(text.strip())
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
        response = llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        messages.append(AIMessage(content=content))

        tool_call = _extract_tool_call(content)
        if not tool_call:
            return content.strip() or "I couldn't process that request."

        name = tool_call.get("tool", "")
        args = tool_call.get("args", {})
        tool_fn = TOOLS.get(name)

        if tool_fn:
            try:
                result = tool_fn.invoke(args)
            except Exception as e:
                result = f"Tool error: {e}"
        else:
            result = f"Unknown tool: {name}"

        messages.append(HumanMessage(content=f"Tool result for {name}:\n{result}"))

    return "I reached the maximum steps. Please try a more specific question."
