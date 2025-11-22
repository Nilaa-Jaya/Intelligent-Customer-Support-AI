from src.database import init_db
from src.main import get_customer_support_agent

# Get agent
agent = get_customer_support_agent()

# Test query
response = agent.process_query(
    query="My app keeps crashing when I try to export data",
    user_id="test_user"
)

# Print results
print(f"\n{'='*60}")
print(f"Query: My app keeps crashing when I try to export data")
print(f"{'='*60}")
print(f"Category: {response['category']}")
print(f"Sentiment: {response['sentiment']}")
print(f"Priority: {response['priority']}/10")
print(f"Processing Time: {response['metadata']['processing_time']:.2f}s")
print(f"\nResponse:\n{response['response']}")
print(f"{'='*60}\n")
