"""
SOP LangGraph Agent — stateful agent graph with tool routing.

Builds a ReAct-style agent using LangGraph's StateGraph,
with conditional routing between the LLM node and the tool node.
"""

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .prompts import SYSTEM_PROMPT
from .tools import setup_tools

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  State definition                                                    #
# ------------------------------------------------------------------ #

class AgentState(TypedDict):
    """State schema for the SOP agent graph."""
    messages: Annotated[list, add_messages]
    current_step: str
    sop_loaded: bool


# ------------------------------------------------------------------ #
#  Graph builder                                                       #
# ------------------------------------------------------------------ #

def create_agent_graph(embedder, tracker):
    """Build and compile the LangGraph agent.

    Args:
        embedder: A SOPEmbedder instance.
        tracker:  A SOPStateTracker instance.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    # 1. Setup tools
    tools = setup_tools(embedder, tracker)

    # 2. Create the LLM with streaming
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    # 3. Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # 4. Create the ToolNode
    tool_node = ToolNode(tools)

    # 5. Agent node — calls the LLM with the system prompt
    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM with the current message history."""
        messages = state["messages"]

        # Ensure the system prompt is at the front
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc)
            response = AIMessage(content=f"⚠️ LLM Error: {exc}. Please retry.")

        return {"messages": [response]}

    # 6. Routing function — continue to tools or end
    def should_continue(state: AgentState) -> str:
        """Decide whether to route to tools or end the turn."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # 7. Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    # 8. Compile and return
    compiled = graph.compile()
    return compiled


# ------------------------------------------------------------------ #
#  Convenience runner                                                  #
# ------------------------------------------------------------------ #

def run_agent(graph, user_message: str, history: list) -> tuple[str, list]:
    """Run the agent graph with a new user message.

    Args:
        graph:        Compiled LangGraph graph.
        user_message: The user's input string.
        history:      List of previous LangChain message objects.

    Returns:
        Tuple of (response_text, updated_messages).
    """
    history = list(history)  # shallow copy to avoid mutation
    history.append(HumanMessage(content=user_message))

    try:
        result = graph.invoke({
            "messages": history,
            "current_step": "",
            "sop_loaded": True,
        })
    except Exception as exc:
        logger.error("Agent invocation error: %s", exc)
        error_msg = f"⚠️ LLM Error: {exc}. Please retry."
        return error_msg, history

    # Extract updated messages and the last AI response
    updated_messages = result.get("messages", history)
    response_text = ""

    # Walk backwards to find the last AIMessage with content
    for msg in reversed(updated_messages):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = msg.content
            break

    if not response_text:
        response_text = "Agent completed processing. No text response generated."

    return response_text, updated_messages
