# Multi-Agent HR Intelligence Platform - Quick Reference Card

## System Architecture at a Glance

```
User Query → Categorizer → Sentiment → KB Retrieval → [Response Agent] → Response
                                            ↓
                                     Escalation Check
```

---

## Agent Pipeline

| Step | Agent | Input | Output | Time |
|------|-------|-------|--------|------|
| 1 | Categorizer | Query | Category | ~300ms |
| 2 | Sentiment Analyzer | Query + Category | Sentiment + Priority | ~300ms |
| 3 | KB Retrieval | Query + Category | Relevant FAQs | ~50ms |
| 4 | Escalation Check | State | should_escalate | ~10ms |
| 5 | Response Agent | Full State | Response | ~500ms |

---

## Categories & Routing

| Category | Keywords | Agent |
|----------|----------|-------|
| Technical | crash, bug, error, slow, sync | `handle_technical()` |
| Billing | charge, refund, payment, invoice | `handle_billing()` |
| Account | login, password, profile, email | `handle_account()` |
| General | (default) | `handle_general()` |

---

## Sentiment → Priority Mapping

| Sentiment | Base Score | Example Query |
|-----------|------------|---------------|
| Positive | 3 | "Thanks for the help!" |
| Neutral | 3 | "How do I export data?" |
| Negative | 5 (+2) | "This keeps failing" |
| Angry | 6 (+3) | "THIS IS UNACCEPTABLE!!!" |

**Modifiers:**
- Repeat Query: +2
- VIP Customer: +2

**Escalation Threshold:** Priority ≥ 8 OR Angry sentiment

---

## Escalation Triggers

| Trigger | Threshold | Example |
|---------|-----------|---------|
| Priority Score | ≥ 8 | Angry + VIP customer |
| Sentiment | Angry | "I DEMAND A REFUND!" |
| Attempt Count | ≥ 3 | Third time asking |
| Keywords | Match | "speak to manager", "lawsuit" |

**Escalation Keywords:**
```
lawsuit, legal, attorney, lawyer, sue, refund immediately,
speak to manager, talk to manager, unacceptable, ridiculous
```

---

## API Endpoints

### Core API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/query` | Process support query |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/stats` | System statistics |

### Webhooks

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/webhooks/` | Create webhook |
| GET | `/api/v1/webhooks/` | List webhooks |
| GET | `/api/v1/webhooks/{id}` | Get webhook |
| PUT | `/api/v1/webhooks/{id}` | Update webhook |
| DELETE | `/api/v1/webhooks/{id}` | Delete webhook |
| POST | `/api/v1/webhooks/{id}/test` | Test webhook |

---

## Request/Response Examples

### Process Query

**Request:**
```json
POST /api/v1/query
{
  "user_id": "user_123",
  "message": "My app keeps crashing"
}
```

**Response:**
```json
{
  "conversation_id": "conv_abc123",
  "response": "I understand how frustrating app crashes can be...",
  "category": "Technical",
  "sentiment": "Neutral",
  "priority": 3,
  "timestamp": "2026-01-16T10:30:00Z",
  "metadata": {
    "processing_time": 0.85,
    "escalated": false,
    "kb_results": [...]
  }
}
```

---

## Database Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `users` | User profiles | user_id, is_vip |
| `conversations` | Query records | query, category, sentiment, response |
| `messages` | Chat history | role, content |
| `feedback` | User ratings | rating (1-5), comment |
| `analytics` | Metrics | counts, averages |
| `knowledge_base` | FAQ storage | title, content, category |
| `webhooks` | Webhook config | url, events, secret_key |
| `webhook_deliveries` | Delivery logs | status, response |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | - | Groq API key |
| `SECRET_KEY` | Yes | - | App secret key |
| `DATABASE_URL` | No | SQLite | PostgreSQL URL |
| `PORT` | No | 8000 | Server port |
| `ENVIRONMENT` | No | development | dev/production |

---

## Key File Locations

```
src/
├── agents/
│   ├── workflow.py      # LangGraph workflow definition
│   ├── state.py         # AgentState TypedDict
│   ├── categorizer.py   # Query classification
│   └── ...              # Other agents
├── api/
│   ├── app.py           # FastAPI application
│   └── routes.py        # API endpoints
├── database/
│   ├── models.py        # SQLAlchemy models
│   └── queries.py       # Query functions
├── knowledge_base/
│   ├── retriever.py     # KB search logic
│   └── vector_store.py  # FAISS implementation
└── utils/
    ├── config.py        # Settings (pydantic)
    └── helpers.py       # Utility functions
```

---

## Common Commands

```bash
# Start development server
python -m uvicorn src.api.app:app --reload --port 8000

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Initialize knowledge base
python initialize_kb.py

# Build Docker image
docker build -t smartsupport-ai .

# Run with Docker Compose
docker-compose up -d

# Deploy to Railway
railway up
```

---

## Debugging Tips

### Check Agent Flow
```python
from src.agents.workflow import get_workflow

workflow = get_workflow()
result = workflow.invoke({"query": "test", "user_id": "test"})
print(result["category"])   # Check categorization
print(result["sentiment"])  # Check sentiment
print(result["kb_results"]) # Check KB retrieval
```

### Test LLM Connection
```python
from src.agents.llm_manager import get_llm_manager

llm = get_llm_manager()
response = llm.invoke_with_retry(
    prompt,
    {"query": "test"}
)
```

### Inspect Vector Store
```python
from src.knowledge_base.retriever import get_kb_retriever

kb = get_kb_retriever()
stats = kb.get_stats()
print(f"Documents: {stats['total_documents']}")
```

---

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Total Response Time | < 2s | 0.8-1.2s |
| Categorization | < 500ms | 200-400ms |
| KB Retrieval | < 100ms | 30-50ms |
| API Availability | 99.9% | - |
| Test Coverage | > 25% | 42% |

---

## Quick Links

- **API Docs:** http://localhost:8000/docs
- **Web UI:** http://localhost:8000/
- **Health Check:** http://localhost:8000/health

---

*Last Updated: January 2026 | Version 2.2.0*
