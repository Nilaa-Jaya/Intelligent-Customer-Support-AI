"""
General and Account support response agents
"""
from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import AgentState
from src.agents.llm_manager import get_llm_manager
from src.utils.logger import app_logger


# General support prompt
GENERAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful customer support agent providing general assistance and information.

Customer Query: {query}

Customer Sentiment: {sentiment}
Priority Level: {priority}

{context}

{kb_context}

Instructions:
1. Provide helpful, accurate information
2. Be friendly and professional
3. Match the customer's emotional tone appropriately
4. Offer additional resources or next steps
5. Keep response concise and clear (150-250 words)

Response:"""
)


# Account support prompt
ACCOUNT_PROMPT = ChatPromptTemplate.from_template(
    """You are an account management and security support agent.

Customer Query: {query}

Customer Sentiment: {sentiment}
Priority Level: {priority}

{context}

{kb_context}

Instructions:
1. Address account-related concerns (login, password, profile, security)
2. Provide clear step-by-step instructions
3. Emphasize security best practices
4. If password reset or security issue, guide through secure process
5. Be reassuring about account security
6. Keep response clear and actionable (200-300 words)

Response:"""
)


def handle_general(state: AgentState) -> AgentState:
    """
    Generate general support response
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with response
    """
    app_logger.info(f"Generating general response for: {state['query'][:50]}...")
    
    try:
        llm_manager = get_llm_manager()
        
        # Prepare conversation context
        context = ""
        if state.get("conversation_history"):
            context = "Previous conversation:\n"
            for msg in state["conversation_history"][-5:]:
                context += f"{msg['role'].capitalize()}: {msg['content']}\n"
            context += "\n"
        
        # Prepare knowledge base context
        kb_context = ""
        if state.get("kb_results"):
            kb_context = "Relevant information:\n"
            for i, kb in enumerate(state["kb_results"][:2], 1):
                kb_context += f"{i}. {kb.get('title', 'N/A')}: {kb.get('content', '')[:200]}...\n"
            kb_context += "\n"
        
        # Invoke LLM
        response = llm_manager.invoke_with_retry(
            GENERAL_PROMPT,
            {
                "query": state["query"],
                "sentiment": state.get("sentiment", "Neutral"),
                "priority": state.get("priority_score", 5),
                "context": context,
                "kb_context": kb_context
            }
        )
        
        app_logger.info("General response generated successfully")
        
        # Update state
        state["response"] = response
        state["next_action"] = "complete"
        
        return state
    
    except Exception as e:
        app_logger.error(f"Error in handle_general: {e}")
        state["response"] = "Thank you for contacting us. How can I assist you today?"
        return state


def handle_account(state: AgentState) -> AgentState:
    """
    Generate account support response
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with response
    """
    app_logger.info(f"Generating account response for: {state['query'][:50]}...")
    
    try:
        llm_manager = get_llm_manager()
        
        # Prepare conversation context
        context = ""
        if state.get("conversation_history"):
            context = "Previous conversation:\n"
            for msg in state["conversation_history"][-5:]:
                context += f"{msg['role'].capitalize()}: {msg['content']}\n"
            context += "\n"
        
        # Prepare knowledge base context
        kb_context = ""
        if state.get("kb_results"):
            kb_context = "Account management resources:\n"
            for i, kb in enumerate(state["kb_results"][:2], 1):
                kb_context += f"{i}. {kb.get('title', 'N/A')}: {kb.get('content', '')[:200]}...\n"
            kb_context += "\n"
        
        # Invoke LLM
        response = llm_manager.invoke_with_retry(
            ACCOUNT_PROMPT,
            {
                "query": state["query"],
                "sentiment": state.get("sentiment", "Neutral"),
                "priority": state.get("priority_score", 5),
                "context": context,
                "kb_context": kb_context
            }
        )
        
        app_logger.info("Account response generated successfully")
        
        # Update state
        state["response"] = response
        state["next_action"] = "complete"
        
        return state
    
    except Exception as e:
        app_logger.error(f"Error in handle_account: {e}")
        state["response"] = "I can help you with your account. Please provide more details about the issue you're experiencing."
        return state
