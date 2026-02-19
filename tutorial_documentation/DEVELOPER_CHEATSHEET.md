# SmartSupport AI - Developer's Cheat Sheet

## Adding a New Agent

### Step 1: Create the Agent File

```python
# src/agents/shipping_agent.py
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState
from src.agents.llm_manager import get_llm_manager
from src.utils.logger import app_logger

SHIPPING_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert shipping and delivery support agent.

Customer Query: {query}
Customer Sentiment: {sentiment}
Priority Level: {priority}

{context}

{kb_context}

Instructions:
1. Address shipping-related concerns
2. Provide tracking information if applicable
3. Explain delivery timeframes
4. Keep response helpful and concise (200-300 words)

Response:"""
)

def handle_shipping(state: AgentState) -> AgentState:
    app_logger.info(f"Generating shipping response: {state['query'][:50]}...")

    try:
        llm_manager = get_llm_manager()

        # Build context (same pattern as other agents)
        context = _build_conversation_context(state)
        kb_context = _build_kb_context(state)

        response = llm_manager.invoke_with_retry(
            SHIPPING_PROMPT,
            {
                "query": state["query"],
                "sentiment": state.get("sentiment", "Neutral"),
                "priority": state.get("priority_score", 5),
                "context": context,
                "kb_context": kb_context,
            },
        )

        state["response"] = response
        state["next_action"] = "complete"
        return state

    except Exception as e:
        app_logger.error(f"Error in handle_shipping: {e}")
        state["response"] = "I apologize, let me connect you with our shipping team."
        return state

def _build_conversation_context(state):
    context = ""
    if state.get("conversation_history"):
        context = "Previous conversation:\n"
        for msg in state["conversation_history"][-5:]:
            context += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return context

def _build_kb_context(state):
    kb_context = ""
    if state.get("kb_results"):
        kb_context = "Relevant shipping information:\n"
        for i, kb in enumerate(state["kb_results"][:2], 1):
            kb_context += f"{i}. {kb.get('title', 'N/A')}: {kb.get('content', '')[:200]}...\n"
    return kb_context
```

### Step 2: Update Categorizer Prompt

```python
# src/agents/categorizer.py - Add to categories
"""
- Shipping: Delivery status, tracking, shipping times, lost packages
"""
```

### Step 3: Update Workflow

```python
# src/agents/workflow.py
from src.agents.shipping_agent import handle_shipping

def create_workflow():
    # ... existing code ...

    # Add new node
    workflow.add_node("shipping", handle_shipping)

    # Update conditional edges
    workflow.add_conditional_edges(
        "check_escalation",
        route_query,
        {
            "technical": "technical",
            "billing": "billing",
            "account": "account",
            "general": "general",
            "shipping": "shipping",  # NEW
            "escalate": "escalate",
        },
    )

    # Add terminal edge
    workflow.add_edge("shipping", END)
```

### Step 4: Update Router

```python
# src/agents/workflow.py
def route_query(state):
    # ... existing code ...

    if category == "Shipping":
        return "shipping"
```

### Step 5: Add FAQs

```json
// data/knowledge_base/faqs.json
{
  "id": 31,
  "category": "Shipping",
  "question": "Where is my package?",
  "answer": "To track your package: 1) Log in to your account..."
}
```

### Step 6: Add Tests

```python
# tests/test_basic.py
def test_shipping_query(agent):
    result = agent.process_query(
        query="Where is my package?",
        user_id="test_user"
    )
    assert result["category"] == "Shipping"
```

---

## Adding a New API Endpoint

### Step 1: Define Schema

```python
# src/api/schemas.py
class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: int  # 1-5
    comment: Optional[str] = None
    was_helpful: bool

class FeedbackResponse(BaseModel):
    id: int
    message: str
```

### Step 2: Create Endpoint

```python
# src/api/routes.py
@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
):
    """Submit feedback for a conversation"""
    try:
        # Get conversation
        conversation = ConversationQueries.get_conversation(
            db, request.conversation_id
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

        # Create feedback
        feedback = FeedbackQueries.create_feedback(
            db=db,
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            rating=request.rating,
            comment=request.comment,
            was_helpful=request.was_helpful,
        )

        return FeedbackResponse(
            id=feedback.id,
            message="Feedback submitted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Adding a New Database Model

### Step 1: Define Model

```python
# src/database/models.py
class Tag(Base):
    """Tag model for organizing conversations"""

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)
    color = Column(String(7))  # Hex color
    created_at = Column(DateTime, default=datetime.utcnow)

    # Many-to-many with conversations
    conversations = relationship(
        "Conversation",
        secondary="conversation_tags",
        back_populates="tags"
    )

# Association table for many-to-many
conversation_tags = Table(
    'conversation_tags',
    Base.metadata,
    Column('conversation_id', ForeignKey('conversations.id')),
    Column('tag_id', ForeignKey('tags.id'))
)
```

### Step 2: Create Migration

```bash
# Generate migration
alembic revision --autogenerate -m "add tags table"

# Run migration
alembic upgrade head
```

### Step 3: Add Query Functions

```python
# src/database/queries.py
class TagQueries:
    @staticmethod
    def create_tag(db: Session, name: str, color: str = None) -> Tag:
        tag = Tag(name=name, color=color)
        db.add(tag)
        db.commit()
        db.refresh(tag)
        return tag

    @staticmethod
    def get_tag_by_name(db: Session, name: str) -> Optional[Tag]:
        return db.query(Tag).filter(Tag.name == name).first()

    @staticmethod
    def add_tag_to_conversation(
        db: Session,
        conversation_id: int,
        tag_id: int
    ):
        conversation = db.query(Conversation).get(conversation_id)
        tag = db.query(Tag).get(tag_id)
        if conversation and tag:
            conversation.tags.append(tag)
            db.commit()
```

---

## Adding a New Webhook Event

### Step 1: Define Event Type

```python
# src/api/webhook_events.py
class WebhookEvents:
    QUERY_CREATED = "query.created"
    QUERY_ESCALATED = "query.escalated"
    FEEDBACK_RECEIVED = "feedback.received"  # NEW

    @classmethod
    def all_events(cls) -> List[str]:
        return [
            cls.QUERY_CREATED,
            cls.QUERY_ESCALATED,
            cls.FEEDBACK_RECEIVED,
        ]
```

### Step 2: Create Payload Factory

```python
# src/api/webhook_events.py
def create_feedback_received_payload(
    webhook_id: str,
    conversation_id: str,
    rating: int,
    comment: Optional[str] = None,
    metadata: Dict = None,
) -> Dict[str, Any]:
    return create_webhook_payload(
        event_type=WebhookEvents.FEEDBACK_RECEIVED,
        webhook_id=webhook_id,
        data={
            "conversation_id": conversation_id,
            "rating": rating,
            "comment": comment,
        },
        metadata=metadata,
    )
```

### Step 3: Trigger Webhook

```python
# In your endpoint
from src.api.webhook_delivery import trigger_webhooks
from src.api.webhook_events import WebhookEvents, create_feedback_received_payload

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # ... create feedback ...

    # Trigger webhook
    payload = create_feedback_received_payload(
        webhook_id="",
        conversation_id=request.conversation_id,
        rating=request.rating,
        comment=request.comment,
    )

    background_tasks.add_task(
        trigger_webhooks,
        db,
        WebhookEvents.FEEDBACK_RECEIVED,
        payload
    )
```

---

## Modifying Escalation Logic

### Current Logic Location
```python
# src/utils/helpers.py
def should_escalate(priority_score, sentiment, attempt_count, query):
    ...
```

### Adding a New Trigger

```python
# Example: Escalate for VIP customers with any negative sentiment

def should_escalate(
    priority_score: int,
    sentiment: str,
    attempt_count: int = 1,
    query: str = "",
    is_vip: bool = False,  # NEW parameter
) -> tuple[bool, Optional[str]]:
    reasons = []

    # Existing triggers...

    # NEW: VIP with negative sentiment
    if is_vip and sentiment in ["Negative", "Angry"]:
        reasons.append("VIP customer with negative sentiment")

    should_escalate_flag = len(reasons) > 0
    return should_escalate_flag, "; ".join(reasons) if reasons else None
```

### Update Escalation Check Agent

```python
# src/agents/escalation_agent.py
def check_escalation(state: AgentState) -> AgentState:
    user_context = state.get("user_context", {})

    needs_escalation, reason = should_escalate(
        priority_score=state.get("priority_score", 5),
        sentiment=state.get("sentiment", "Neutral"),
        attempt_count=user_context.get("attempt_count", 1),
        query=state["query"],
        is_vip=user_context.get("is_vip", False),  # Pass VIP status
    )

    state["should_escalate"] = needs_escalation
    state["escalation_reason"] = reason
    return state
```

---

## Modifying Knowledge Base

### Adding FAQs

```python
# Script to add FAQs programmatically
from src.knowledge_base.retriever import get_kb_retriever

def add_new_faq():
    import json

    # Load existing FAQs
    with open("data/knowledge_base/faqs.json", "r") as f:
        data = json.load(f)

    # Add new FAQ
    new_faq = {
        "id": len(data["faqs"]) + 1,
        "category": "Technical",
        "question": "How do I clear my cache?",
        "answer": "To clear cache: 1) Go to Settings..."
    }
    data["faqs"].append(new_faq)

    # Save
    with open("data/knowledge_base/faqs.json", "w") as f:
        json.dump(data, f, indent=2)

    # Rebuild index
    kb = get_kb_retriever(force_reload=True)
    print(f"KB now has {kb.get_stats()['total_documents']} documents")
```

### Updating Similarity Threshold

```python
# src/agents/kb_retrieval.py
results = kb_retriever.retrieve(
    query=query,
    k=3,
    category=category,
    min_score=0.25,  # Lower threshold for more results
)
```

---

## Common Testing Patterns

### Testing an Agent in Isolation

```python
import pytest
from src.agents.state import AgentState
from src.agents.categorizer import categorize_query

def test_categorize_technical():
    state = AgentState(
        query="My app crashes when I click save",
        user_id="test",
        conversation_id="test_conv",
        # ... other required fields with defaults
        category=None,
        sentiment=None,
        priority_score=None,
        kb_results=None,
        response=None,
        should_escalate=False,
        escalation_reason=None,
        next_action=None,
        user_context={},
        conversation_history=[],
        metadata={},
        processing_time=None,
        user_db_id=None,
        conversation_db_id=None,
    )

    result = categorize_query(state)
    assert result["category"] == "Technical"
```

### Testing Full Workflow

```python
def test_full_workflow():
    from src.agents.workflow import get_workflow
    from src.agents.state import ConversationContext

    context = ConversationContext(
        query="How do I reset my password?",
        user_id="test_user",
        conversation_id="test_conv",
    )

    workflow = get_workflow()
    result = workflow.invoke(context.to_state())

    assert result["category"] == "Account"
    assert "password" in result["response"].lower()
```

### Mocking LLM for Faster Tests

```python
from unittest.mock import patch, MagicMock

def test_categorizer_with_mock():
    mock_llm = MagicMock()
    mock_llm.invoke_with_retry.return_value = "Technical"

    with patch('src.agents.categorizer.get_llm_manager', return_value=mock_llm):
        state = {"query": "test", "user_id": "test", ...}
        result = categorize_query(state)
        assert result["category"] == "Technical"
```

---

## Debugging Tips

### Enable Debug Logging

```python
# src/utils/logger.py
import logging
from loguru import logger

# Set to DEBUG for verbose output
logger.add(
    "logs/debug.log",
    level="DEBUG",
    rotation="1 day",
)
```

### Inspect Workflow State

```python
# Add this to any agent for debugging
def handle_technical(state: AgentState) -> AgentState:
    import json
    print("=" * 50)
    print("STATE AT TECHNICAL AGENT:")
    print(json.dumps({k: str(v)[:100] for k, v in state.items()}, indent=2))
    print("=" * 50)

    # ... rest of function
```

### Test KB Retrieval Directly

```python
from src.knowledge_base.retriever import get_kb_retriever

kb = get_kb_retriever()

# Test search
results = kb.retrieve("how to reset password", k=5)
for r in results:
    print(f"[{r['score']:.3f}] {r['question'][:50]}...")
```

---

## Performance Optimization Tips

### 1. Cache LLM Responses
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_categorize(query: str) -> str:
    # Only cache deterministic queries
    ...
```

### 2. Batch KB Loading
```python
# Load KB once at startup, not per-request
kb_retriever = None

@app.on_event("startup")
async def startup():
    global kb_retriever
    kb_retriever = get_kb_retriever()
```

### 3. Async Database Operations
```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# Use async for non-blocking DB calls
async with AsyncSession(engine) as session:
    result = await session.execute(query)
```

---

## Quick Copy-Paste Templates

### New Agent Template
```python
"""
[AGENT_NAME] Agent
"""
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState
from src.agents.llm_manager import get_llm_manager
from src.utils.logger import app_logger

[AGENT_NAME]_PROMPT = ChatPromptTemplate.from_template("""...""")

def handle_[agent_name](state: AgentState) -> AgentState:
    try:
        llm_manager = get_llm_manager()
        response = llm_manager.invoke_with_retry(...)
        state["response"] = response
        return state
    except Exception as e:
        app_logger.error(f"Error: {e}")
        state["response"] = "..."
        return state
```

### New Endpoint Template
```python
@router.post("/[endpoint]", response_model=[Response]Model)
async def [endpoint_name](
    request: [Request]Model,
    db: Session = Depends(get_db),
):
    try:
        # Business logic
        return [Response]Model(...)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### New Test Template
```python
import pytest
from src.main import get_customer_support_agent

@pytest.fixture
def agent():
    return get_customer_support_agent()

def test_[test_name](agent):
    result = agent.process_query(
        query="...",
        user_id="test_user"
    )
    assert result["..."] == "..."
```

---

*Last Updated: January 2026 | SmartSupport AI v2.2.0*
