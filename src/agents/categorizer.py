"""
Query categorization agent
"""

from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import AgentState
from src.agents.llm_manager import get_llm_manager
from src.utils.helpers import parse_llm_category
from src.utils.logger import app_logger


# Categorization prompt
CATEGORIZATION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert customer support query classifier.

Categorize the following customer query into ONE of these categories:
- Technical: Issues with software, hardware, service functionality, bugs, errors, setup, configuration
- Billing: Payment issues, invoices, refunds, subscriptions, pricing, charges
- Account: Login, password, profile, account settings, registration, security
- General: Company policies, general inquiries, feedback, suggestions

Query: {query}

{context}

Respond with ONLY the category name (Technical, Billing, Account, or General).
Category:"""
)


def categorize_query(state: AgentState) -> AgentState:
    """
    Categorize customer query

    Args:
        state: Current agent state

    Returns:
        Updated state with category
    """
    app_logger.info(f"Categorizing query: {state['query'][:50]}...")

    try:
        llm_manager = get_llm_manager()

        # Prepare context
        context = ""
        if state.get("conversation_history"):
            context = "Previous conversation context:\n"
            for msg in state["conversation_history"][-5:]:
                context += f"{msg['role']}: {msg['content'][:150]}\n"

        # Invoke LLM
        raw_category = llm_manager.invoke_with_retry(
            CATEGORIZATION_PROMPT, {"query": state["query"], "context": context}
        )

        # Parse and standardize category
        category = parse_llm_category(raw_category)

        app_logger.info(f"Query categorized as: {category}")

        # Update state
        state["category"] = category

        # Update metadata
        if not state.get("extra_metadata"):
            state["extra_metadata"] = {}
        state["extra_metadata"]["raw_category"] = raw_category

        return state

    except Exception as e:
        app_logger.error(f"Error in categorize_query: {e}")
        # Fallback to General category
        state["category"] = "General"
        return state
