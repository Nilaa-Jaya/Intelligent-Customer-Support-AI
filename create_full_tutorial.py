"""
Create comprehensive tutorial with all missing sections
"""

# Read the original
with open('tutorial_documentation/SMARTSUPPORT_AI_COMPLETE_TUTORIAL.md', 'r', encoding='utf-8') as f:
    original = f.read()

# Find where to insert new sections
ch17_end_pos = original.find('\n---\n\n## Chapter 18: API Implementation')
ch18_start_pos = ch17_end_pos + len('\n---\n\n')

# Extract everything before Chapter 18
before_ch18 = original[:ch17_end_pos]

# Extract everything from Chapter 18 onward
from_ch18 = original[ch18_start_pos:]

# Now I'll build the new content with missing sections

# Section 17.4: Error Handling
section_17_4 = """

### 17.4 Error Handling

Error handling in API applications is critical for reliability and user experience. A well-designed error handling system provides clear feedback, maintains security, and aids debugging.

**Theory: Why Error Handling Matters**

Think of error handling like defensive driving - you don't just plan for things to go right, you plan for when things go wrong. In production systems, errors WILL occur:
- External API failures (Groq API down, rate limits)
- Database connection issues
- Invalid user input
- Unexpected data formats

Good error handling turns catastrophic failures into recoverable situations.

**HTTP Status Codes and When to Use Them**

HTTP status codes communicate what happened with a request. Here's when to use each:

| Code | Name | When to Use | Example |
|------|------|-------------|---------|
| 200 | OK | Successful request | Query processed successfully |
| 201 | Created | Resource created | New webhook registered |
| 400 | Bad Request | Client sent invalid data | Missing required field |
| 401 | Unauthorized | Authentication required | Missing or invalid API key |
| 403 | Forbidden | User lacks permission | VIP-only feature accessed by regular user |
| 404 | Not Found | Resource doesn't exist | Webhook ID not in database |
| 422 | Unprocessable Entity | Valid format, invalid data | Email format correct but domain invalid |
| 429 | Too Many Requests | Rate limit exceeded | More than 100 requests/minute |
| 500 | Internal Server Error | Server-side error | Database connection failed |
| 503 | Service Unavailable | Temporary outage | LLM API is down |

> **KEY INSIGHT:** Use 4xx codes for client errors (user's fault), 5xx codes for server errors (your fault). This distinction helps with debugging and monitoring.

**Error Response Schema**

Consistent error responses make your API predictable and easy to use:

```python
# From src/api/schemas.py
from pydantic import BaseModel
from typing import Optional, List

class ErrorDetail(BaseModel):
    """Detailed error information"""
    field: Optional[str] = None  # Which field caused the error
    message: str                  # Human-readable error message
    code: Optional[str] = None    # Machine-readable error code

class ErrorResponse(BaseModel):
    """Standard error response format"""
    error: str                    # Error category
    message: str                  # User-friendly message
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None  # For support/debugging

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data provided",
                "details": [
                    {
                        "field": "user_id",
                        "message": "user_id cannot be empty",
                        "code": "REQUIRED_FIELD"
                    }
                ],
                "request_id": "req_abc123"
            }
        }
```

**FastAPI Exception Handlers**

FastAPI provides a powerful exception handling system. Here's our implementation:

```python
# From src/api/app.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
import httpx
import uuid

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors (400 Bad Request)

    When user sends invalid data (missing fields, wrong types),
    this handler formats the error into our standard response.
    """
    details = []
    for error in exc.errors():
        details.append({
            "field": ".".join(str(loc) for loc in error["loc"][1:]),  # Skip 'body'
            "message": error["msg"],
            "code": error["type"].upper()
        })

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": details,
            "request_id": str(uuid.uuid4())
        }
    )

@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    """
    Handle database errors (500 Internal Server Error)

    Database errors are internal issues - don't expose details to users.
    """
    app_logger.error(f"Database error: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "DatabaseError",
            "message": "An internal database error occurred. Please try again later.",
            "request_id": str(uuid.uuid4())
        }
    )

@app.exception_handler(httpx.HTTPStatusError)
async def llm_api_exception_handler(request: Request, exc: httpx.HTTPStatusError):
    """
    Handle LLM API errors (503 Service Unavailable)

    When Groq API fails, we don't want to crash - gracefully handle it.
    """
    app_logger.error(f"LLM API error: {exc}")

    # Check if it's a rate limit error
    if exc.response.status_code == 429:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "RateLimitError",
                "message": "AI service rate limit exceeded. Please try again in a moment.",
                "request_id": str(uuid.uuid4())
            }
        )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "ServiceUnavailable",
            "message": "AI service temporarily unavailable. Please try again later.",
            "request_id": str(uuid.uuid4())
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected errors (500 Internal Server Error)

    This is the safety net - catches anything we didn't anticipate.
    """
    app_logger.error(f"Unexpected error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalError",
            "message": "An unexpected error occurred. Our team has been notified.",
            "request_id": str(uuid.uuid4())
        }
    )
```

**Why This Design Works:**

1. **Specificity:** Different exception types get different handlers
2. **Security:** Never expose internal details (stack traces, DB queries) to users
3. **Debugging:** Log full details server-side, show safe messages to users
4. **Traceability:** request_id allows support to find the exact error in logs

**Retry Logic and Graceful Degradation**

When external dependencies fail, don't fail immediately - retry with backoff:

```python
# From src/agents/llm_manager.py
import time
from typing import Optional

class LLMManager:
    def invoke_with_retry(
        self,
        prompt_template,
        variables: dict,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> str:
        """
        Invoke LLM with exponential backoff retry logic

        Args:
            prompt_template: ChatPromptTemplate to use
            variables: Variables to fill in template
            max_retries: Maximum number of attempts
            base_delay: Initial delay in seconds (doubles each retry)

        Returns:
            LLM response string

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                # Create prompt chain
                chain = prompt_template | self.llm

                # Invoke LLM
                response = chain.invoke(variables)

                # Extract text content
                if hasattr(response, 'content'):
                    return response.content.strip()
                return str(response).strip()

            except httpx.HTTPStatusError as e:
                last_exception = e

                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    app_logger.warning(f"LLM API client error: {e}")
                    raise

                # Retry on server errors (5xx) and network issues
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    app_logger.warning(
                        f"LLM API error on attempt {attempt}/{max_retries}. "
                        f"Retrying in {delay}s... Error: {e}"
                    )
                    time.sleep(delay)
                else:
                    app_logger.error(f"LLM API error after {max_retries} attempts: {e}")
                    raise

            except Exception as e:
                last_exception = e
                app_logger.error(f"Unexpected error in LLM invocation: {e}")
                raise

        # Should never reach here, but just in case
        raise last_exception or Exception("All retries exhausted")
```

**Retry Strategy Explained:**

- **Attempt 1:** Immediate (delay = 0s)
- **Attempt 2:** After 1 second (delay = 1s)
- **Attempt 3:** After 2 seconds (delay = 2s)
- **Attempt 4:** After 4 seconds (delay = 4s)

This exponential backoff prevents overwhelming a recovering service.

**Graceful Degradation Example:**

```python
# From src/agents/categorizer.py
def categorize_query(state: AgentState) -> AgentState:
    """Categorize with fallback to rule-based system"""
    try:
        # Try AI categorization
        category = ai_categorize(state["query"])
        state["category"] = category
        state["categorization_method"] = "ai"

    except Exception as e:
        app_logger.warning(f"AI categorization failed, using rules: {e}")

        # Fallback to keyword-based rules
        query_lower = state["query"].lower()

        if any(word in query_lower for word in ["crash", "bug", "error", "not working"]):
            category = "Technical"
        elif any(word in query_lower for word in ["charge", "payment", "refund", "bill"]):
            category = "Billing"
        elif any(word in query_lower for word in ["login", "password", "account"]):
            category = "Account"
        else:
            category = "General"

        state["category"] = category
        state["categorization_method"] = "rules_fallback"

    return state
```

> **KEY INSIGHT:** Graceful degradation means "degrade service quality, not availability." A rule-based categorization (less accurate) is better than no categorization (total failure).

**Example Error Scenarios and Responses:**

**Scenario 1: Missing Required Field**
```bash
curl -X POST http://localhost:8000/api/v1/query \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Help me"}'  # Missing user_id

# Response: 400 Bad Request
{
  "error": "ValidationError",
  "message": "Invalid request data",
  "details": [
    {
      "field": "user_id",
      "message": "field required",
      "code": "VALUE_ERROR.MISSING"
    }
  ],
  "request_id": "req_7a8b9c"
}
```

**Scenario 2: LLM API Rate Limit**
```python
# Internal: Groq returns 429
# Our handler catches it and returns:

# Response: 429 Too Many Requests
{
  "error": "RateLimitError",
  "message": "AI service rate limit exceeded. Please try again in a moment.",
  "request_id": "req_1d2e3f"
}
```

**Scenario 3: Database Connection Lost**
```python
# Internal: SQLAlchemy raises OperationalError
# Our handler catches it and returns:

# Response: 500 Internal Server Error
{
  "error": "DatabaseError",
  "message": "An internal database error occurred. Please try again later.",
  "request_id": "req_4g5h6i"
}
```

> **WARNING:** Never return stack traces or internal error details in production. They expose system architecture and can aid attackers.

**Testing Error Handlers:**

```python
# From tests/test_error_handling.py
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_missing_required_field():
    """Test validation error handling"""
    response = client.post("/api/v1/query", json={"message": "test"})

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "ValidationError"
    assert "user_id" in str(data["details"])

def test_webhook_not_found():
    """Test 404 error handling"""
    response = client.get("/api/v1/webhooks/nonexistent_id")

    assert response.status_code == 404
    data = response.json()
    assert data["error"] == "NotFound"

@pytest.mark.integration
def test_llm_api_failure_retry():
    """Test LLM API failure with retry (requires mocking)"""
    # Mock Groq API to fail twice, succeed third time
    # Verify retries happen with exponential backoff
    pass
```

**Best Practices Summary:**

1. **Use Standard HTTP Status Codes:** Don't invent your own error system
2. **Consistent Response Format:** All errors follow same JSON structure
3. **Security First:** Hide internal details from users
4. **Comprehensive Logging:** Log everything server-side with context
5. **Retry with Backoff:** Don't immediately fail on transient errors
6. **Graceful Degradation:** Provide reduced functionality rather than none
7. **Include request_id:** Essential for debugging and support
8. **Test Error Paths:** Write tests for error scenarios, not just happy path

---
"""

print("Building new tutorial with all missing sections...")
print(f"Original length: {len(original)} characters")
print(f"Before Chapter 18: {len(before_ch18)} characters")
print(f"Section 17.4 length: {len(section_17_4)} characters")

# For now, let's just show what we would add
print("\n=== NEW SECTION 17.4 would be added ===")
print(section_17_4[:500] + "...")
