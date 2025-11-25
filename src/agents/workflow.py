"""
Main workflow orchestrator using LangGraph
"""

from langgraph.graph import StateGraph, END
from typing import Literal

from src.agents.state import AgentState
from src.agents.categorizer import categorize_query
from src.agents.sentiment_analyzer import analyze_sentiment
from src.agents.kb_retrieval import retrieve_from_kb
from src.agents.technical_agent import handle_technical
from src.agents.billing_agent import handle_billing
from src.agents.general_agent import handle_general, handle_account
from src.agents.escalation_agent import check_escalation, escalate_to_human
from src.utils.logger import app_logger


def route_query(
    state: AgentState,
) -> Literal["escalate", "technical", "billing", "account", "general"]:
    """
    Route query based on escalation check and category

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    # First check if escalation is needed
    if state.get("should_escalate", False):
        app_logger.info("Routing to escalation")
        return "escalate"

    # Route based on category
    category = state.get("category", "General")

    if category == "Technical":
        app_logger.info("Routing to technical agent")
        return "technical"
    elif category == "Billing":
        app_logger.info("Routing to billing agent")
        return "billing"
    elif category == "Account":
        app_logger.info("Routing to account agent")
        return "account"
    else:
        app_logger.info("Routing to general agent")
        return "general"


def create_workflow() -> StateGraph:
    """
    Create the customer support workflow graph

    Returns:
        Compiled StateGraph workflow
    """
    app_logger.info("Creating customer support workflow...")

    # Initialize workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("categorize", categorize_query)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("retrieve_kb", retrieve_from_kb)
    workflow.add_node("check_escalation", check_escalation)
    workflow.add_node("technical", handle_technical)
    workflow.add_node("billing", handle_billing)
    workflow.add_node("account", handle_account)
    workflow.add_node("general", handle_general)
    workflow.add_node("escalate", escalate_to_human)

    # Set entry point
    workflow.set_entry_point("categorize")

    # Add edges
    workflow.add_edge("categorize", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "retrieve_kb")
    workflow.add_edge("retrieve_kb", "check_escalation")

    # Add conditional routing after escalation check
    workflow.add_conditional_edges(
        "check_escalation",
        route_query,
        {
            "technical": "technical",
            "billing": "billing",
            "account": "account",
            "general": "general",
            "escalate": "escalate",
        },
    )

    # All response nodes lead to END
    workflow.add_edge("technical", END)
    workflow.add_edge("billing", END)
    workflow.add_edge("account", END)
    workflow.add_edge("general", END)
    workflow.add_edge("escalate", END)

    # Compile workflow
    app_logger.info("Workflow created successfully")
    return workflow.compile()


# Global workflow instance
_workflow = None


def get_workflow():
    """Get or create workflow singleton"""
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
    return _workflow
