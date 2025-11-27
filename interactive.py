"""
Interactive CLI for SmartSupport AI
Test the customer support agent interactively
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import init_db
from src.main import get_customer_support_agent


def print_header():
    """Print welcome header"""
    print("\n" + "=" * 80)
    print(" SmartSupport AI - Interactive Customer Support Agent")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type your query to get support")
    print("  - 'history' - View conversation history")
    print("  - 'stats' - View session statistics")
    print("  - 'clear' - Clear screen")
    print("  - 'quit' or 'exit' - Exit the program")
    print("\n" + "-" * 80 + "\n")


def print_response(response):
    """Pretty print response"""
    print("\n" + "─" * 80)
    print(f" Category: {response['category']}")
    print(f"😊 Sentiment: {response['sentiment']}")
    print(f"* Priority: {response['priority']}/10")
    print(f"  Processing Time: {response['metadata']['processing_time']:.3f}s")

    if response["metadata"].get("escalated"):
        print(
            f"🚨 Escalated: Yes - {response['metadata'].get('escalation_reason', 'N/A')}"
        )

    print("\n Response:")
    print(response["response"])
    print("─" * 80 + "\n")


def print_history(history):
    """Print conversation history"""
    if not history:
        print("\n📭 No conversation history found.\n")
        return

    print("\n" + "─" * 80)
    print(f" Conversation History ({len(history)} conversations)")
    print("─" * 80)

    for i, conv in enumerate(history, 1):
        status = "🚨 ESCALATED" if conv["escalated"] else "[DONE] RESOLVED"
        print(f"\n{i}. [{conv['category']}] {status}")
        print(f"   Query: {conv['query'][:70]}...")
        print(f"   Time: {conv['timestamp']}")

    print("─" * 80 + "\n")


def main():
    """Main interactive loop"""

    print_header()

    # Initialize
    print("Initializing system...")
    try:
        init_db()
        agent = get_customer_support_agent()
        print("[DONE] System ready!\n")
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        return

    # Get user ID
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"

    print(f"\n👤 User: {user_id}\n")
    print("Type your support query below:\n")

    query_count = 0

    # Interactive loop
    while True:
        try:
            # Get input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 Thank you for using SmartSupport AI. Goodbye!\n")
                break

            elif user_input.lower() == "history":
                history = agent.get_conversation_history(user_id, limit=10)
                print_history(history)
                continue

            elif user_input.lower() == "stats":
                print("\n" + "─" * 80)
                print(f" Session Statistics")
                print("─" * 80)
                print(f"User ID: {user_id}")
                print(f"Queries in this session: {query_count}")
                history = agent.get_conversation_history(user_id, limit=100)
                print(f"Total conversations: {len(history)}")
                print("─" * 80 + "\n")
                continue

            elif user_input.lower() == "clear":
                import os

                os.system("clear" if os.name != "nt" else "cls")
                print_header()
                continue

            # Process query
            print("\n⏳ Processing your query...\n")

            response = agent.process_query(query=user_input, user_id=user_id)

            query_count += 1

            # Display response
            print_response(response)

        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using SmartSupport AI. Goodbye!\n")
            break

        except Exception as e:
            print(f"\n[FAIL] Error: {e}\n")
            continue


if __name__ == "__main__":
    main()
