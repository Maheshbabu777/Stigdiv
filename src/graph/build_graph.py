from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import general_node, market_node, news_node, recall_node, save_node, social_node, supervisor_node
from src.graph.router import next_nodes, route_message
from src.graph.state import BriefingState


def build_graph():
    graph = StateGraph(BriefingState)
    graph.add_node("router", route_message)
    graph.add_node("news", news_node)
    graph.add_node("market", market_node)
    graph.add_node("social", social_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("save", save_node)
    graph.add_node("recall", recall_node)
    graph.add_node("general", general_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", next_nodes)
    graph.add_edge("news", "supervisor")
    graph.add_edge("market", "supervisor")
    graph.add_edge("social", "supervisor")
    graph.add_edge("supervisor", "save")
    graph.add_edge("save", END)
    graph.add_edge("recall", END)
    graph.add_edge("general", END)
    return graph.compile()
