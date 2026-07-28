import os
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

MAX_ITERATIONS = 3


class FlowState(TypedDict):
    topic: str
    research: str
    summary: str
    critique: str
    approved: bool
    iteration: int


def route_after_critique(state: FlowState) -> str:
    if state["approved"] or state["iteration"] >= MAX_ITERATIONS:
        return END
    return "summarize"


def build_graph():
    llm = ChatOllama(
        model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    def research_node(state: FlowState) -> dict:
        response = llm.invoke(
            f"List the key facts and points someone should know about: {state['topic']}"
        )
        return {"research": response.content}

    def summarize_node(state: FlowState) -> dict:
        if state["iteration"] == 0:
            prompt = f"Summarize the following research in a short paragraph:\n\n{state['research']}"
        else:
            prompt = (
                f"Revise this summary based on the critique below.\n\n"
                f"Summary:\n{state['summary']}\n\nCritique:\n{state['critique']}"
            )
        response = llm.invoke(prompt)
        return {"summary": response.content}

    def critique_node(state: FlowState) -> dict:
        response = llm.invoke(
            "Review this summary for accuracy and completeness. "
            "Reply with 'APPROVED' on the first line if it's good enough, "
            "otherwise reply 'REVISE' followed by what's missing.\n\n"
            f"Summary:\n{state['summary']}"
        )
        text = response.content
        approved = text.strip().upper().startswith("APPROVED")
        return {
            "critique": text,
            "approved": approved,
            "iteration": state["iteration"] + 1,
        }

    graph = StateGraph(FlowState)
    graph.add_node("research", research_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("critique", critique_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", "critique")
    graph.add_conditional_edges("critique", route_after_critique)

    return graph.compile()
