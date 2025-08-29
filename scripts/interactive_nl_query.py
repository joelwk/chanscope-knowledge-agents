#!/usr/bin/env python
"""
Interactive Natural Language Query Tool

This script provides an interactive console interface for sending natural
language queries to the Chanscope nl_query API endpoint. The server converts
NL → SQL (via the Venice SQL "character"), executes it against PostgreSQL,
and returns semantically relevant rows.
"""

import os
import requests
import json
import sys
from datetime import datetime
import argparse


def _build_endpoint(api_url: str) -> str:
    """Normalize API base and produce the full nl_query endpoint.

    Accepts any of the following forms and resolves to the correct endpoint:
      - http://host                     → http://host/api/v1/nl_query
      - http://host:8080                → http://host:8080/api/v1/nl_query
      - http://host/api                 → http://host/api/v1/nl_query
      - http://host/api/v1              → http://host/api/v1/nl_query
      - http://host/api/v1/             → http://host/api/v1/nl_query
      - http://host/api/v1/nl_query     → http://host/api/v1/nl_query
    """
    base = (api_url or "").strip().rstrip("/")
    if not base:
        base = "http://localhost:8080"  # sensible default

    # If caller already supplied the full endpoint, return as-is
    if base.endswith("/api/v1/nl_query"):
        return base

    if base.endswith("/api/v1"):
        return f"{base}/nl_query"
    if base.endswith("/api"):
        return f"{base}/v1/nl_query"
    # Otherwise assume host root
    return f"{base}/api/v1/nl_query"


def send_query(api_url: str, query: str, limit: int = 10, provider: str = None):
    """
    Send a natural language query to the API and return the response.
    
    Args:
        api_url: Base URL of the API
        query: Natural language query string
        limit: Maximum number of results to return
        provider: Optional model provider (openai, grok, venice)
        
    Returns:
        API response as dictionary or None if error
    """
    endpoint = _build_endpoint(api_url)
    
    # Prepare the payload
    payload = {
        "query": query,
        "limit": limit
    }
    
    if provider:
        payload["provider"] = provider
    
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    try:
        print(f"\n📤 Sending query to {endpoint}...")
        response = requests.post(endpoint, json=payload, headers=headers)
        
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                print("❌ Error: Server returned non-JSON content.")
                print(response.text[:500])
                return None
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            try:
                err = response.json()
                if isinstance(err, dict):
                    # Friendly display for FastAPI error format
                    detail = err.get("detail") or err
                    if isinstance(detail, dict):
                        msg = detail.get("message") or detail.get("error") or str(detail)
                        print(f"Reason: {msg}")
                    else:
                        print(detail)
                else:
                    print(err)
            except Exception:
                print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to {endpoint}")
        print("Make sure the API server is running.")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def display_results(response: dict):
    """
    Display the query results in a formatted manner.
    
    Args:
        response: API response dictionary
    """
    if not response:
        return
    
    print("\n" + "="*80)
    print("QUERY RESULTS")
    print("="*80)
    
    # Display query info
    print(f"\n📝 Original Query: {response.get('query', 'N/A')}")
    print(f"🔍 Generated SQL: {response.get('sql', 'N/A')}")
    print(f"⏱️  Execution Time: {response.get('execution_time_ms', 'N/A')} ms")
    print(f"📊 Records Found: {response.get('record_count', 0)}")
    
    # Display filters applied
    if 'description' in response and 'filters' in response['description']:
        print("\n🔧 Filters Applied:")
        for filter_desc in response['description']['filters']:
            print(f"   • {filter_desc}")
    
    # Display results in a table
    data = response.get('data', [])
    if data:
        print(f"\n📋 Results (showing up to {len(data)} records):")

        # Determine columns dynamically based on response
        preferred_cols = [
            "id", "thread_id", "posted_date_time", "content", "board", "channel_name", "author"
        ]
        # Collect keys present across first few rows
        keys_seen = []
        for rec in data[:5]:
            for k in rec.keys():
                if k not in keys_seen:
                    keys_seen.append(k)
        # Build ordered display columns: preferred first if present, then others
        cols = [c for c in preferred_cols if c in keys_seen]
        cols += [k for k in keys_seen if k not in cols]
        # Always include at least content/posted_date_time if available
        if not cols and data and isinstance(data[0], dict):
            cols = list(data[0].keys())

        # Prepare table data with truncation for long text
        def _shorten(val, width=80):
            s = str(val) if val is not None else ""
            return (s[: width - 3] + "...") if len(s) > width else s

        rows = []
        for rec in data[:10]:
            rows.append([_shorten(rec.get(c, ""), 80 if c == "content" else 30) for c in cols])

        # Try to use tabulate if available, otherwise fallback
        try:
            from tabulate import tabulate  # lazy import; optional dependency

            print(tabulate(rows, headers=cols, tablefmt="grid"))
        except Exception:
            # Simple fallback table
            print(" | ".join(cols))
            print("-" * 80)
            for r in rows:
                print(" | ".join(map(str, r)))

        if len(data) > 10:
            print(f"\n... and {len(data) - 10} more records")
    else:
        print("\n📭 No results found.")
    
    print("\n" + "="*80)


def interactive_mode(api_url: str, default_limit: int = 10, provider: str = None):
    """
    Run the interactive query mode.
    
    Args:
        api_url: Base URL of the API
        default_limit: Default result limit
        provider: Optional model provider
    """
    print("\n🚀 Chanscope Natural Language Query Interface")
    print("="*50)
    print("Enter your queries in natural language.")
    print("Type 'help' for examples, 'quit' to exit.")
    print("="*50)
    
    while True:
        try:
            # Get user input
            print("\n💭 Enter your query (or 'quit' to exit):")
            user_input = input(">>> ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            elif user_input.lower() == 'help':
                print("\n📚 Example queries:")
                print("  • Show me posts from the last hour")
                print("  • Find messages containing bitcoin from yesterday")
                print("  • Get 5 recent posts about AI")
                print("  • Show posts from channel tech in the last 3 days")
                print("  • Find posts by author satoshi containing crypto")
                print("  • What are people saying about market trends?")
                continue
                
            elif user_input.lower() == 'clear':
                print("\033[2J\033[H")  # Clear screen
                continue
                
            elif not user_input:
                continue
            
            # Check if user wants to specify a custom limit
            limit = default_limit
            if "limit" in user_input.lower():
                # Try to extract number before "limit"
                import re
                match = re.search(r'(\d+)\s*(?:limit|results?|rows?)', user_input.lower())
                if match:
                    limit = int(match.group(1))
            
            # Send the query
            response = send_query(api_url, user_input, limit, provider)
            
            # Display results
            display_results(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Interactive Natural Language Query Tool for Chanscope"
    )
    parser.add_argument(
        "--api",
        default=os.environ.get("API_BASE_URL", "http://localhost:8080/api/v1"),
        help="API base URL or full endpoint. Examples: http://localhost:8080, http://host/api/v1, or full /api/v1/nl_query"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Default number of results to return (default: 10)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "grok", "venice"],
        help="Model provider to use for query processing"
    )
    parser.add_argument(
        "--query",
        help="Single query to execute (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Check if running in single query mode
    if args.query:
        print(f"\n🔍 Executing query: {args.query}")
        response = send_query(args.api, args.query, args.limit, args.provider)
        display_results(response)
    else:
        # Run interactive mode
        interactive_mode(args.api, args.limit, args.provider)


if __name__ == "__main__":
    main()
