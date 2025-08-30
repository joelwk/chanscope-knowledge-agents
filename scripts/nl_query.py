#!/usr/bin/env python
"""
Minimal Natural Language → SQL client for the Chanscope API.

- Designed for Replit workflow usage to query the Replit Postgres DB via
  the existing `/api/v1/nl_query` endpoint.
- Works in single-shot mode (positional query) or interactive REPL.
"""

import os
import json
import argparse
from typing import Optional, Dict, Any, List

import requests


def _build_endpoint(api_base: str) -> str:
    """Normalize API base into the full /api/v1/nl_query endpoint."""
    base = (api_base or "").strip().rstrip("/") or "http://localhost"
    if base.endswith("/api/v1/nl_query"):
        return base
    if base.endswith("/api/v1"):
        return f"{base}/nl_query"
    if base.endswith("/api"):
        return f"{base}/v1/nl_query"
    return f"{base}/api/v1/nl_query"


def _default_api_base() -> str:
    """Best-effort default API base for Replit/local.

    Priority:
    - API_BASE_URL (use as-is)
    - API_PORT -> http://localhost:<port>
    - fallback http://localhost
    """
    env_base = os.environ.get("API_BASE_URL")
    if env_base:
        return env_base
    port = os.environ.get("API_PORT")
    if port:
        return f"http://localhost:{port}"
    return "http://localhost"


def send_query(api_base: str, query: str, limit: int = 10) -> Optional[Dict[str, Any]]:
    """Send NL query to API and return parsed JSON or None on error."""
    endpoint = _build_endpoint(api_base)
    payload = {"query": query, "limit": int(limit)}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    try:
        print(f"\n-> Sending query to {endpoint}")
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()

        print(f"ERROR: HTTP {resp.status_code}")
        try:
            data = resp.json()
        except Exception:
            print((resp.text or "").strip()[:500])
            return None

        detail = data.get("detail", data)
        if isinstance(detail, dict):
            code = detail.get("error") or detail.get("code")
            msg = detail.get("message") or detail.get("detail") or str(detail)
            if code == "ENVIRONMENT_RESTRICTION":
                print("This endpoint only works in Replit (PostgreSQL enabled).")
            print(f"Reason: {msg}")
        else:
            print(str(detail))
        return None

    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to {endpoint}. Is the API running?")
        return None
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out.")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def display_results(response: Dict[str, Any], as_json: bool = False) -> None:
    """Pretty-print results. If as_json, dump raw JSON."""
    if not response:
        return

    if as_json:
        print(json.dumps(response, indent=2, default=str))
        return

    print("\n" + "=" * 80)
    print("QUERY RESULTS")
    print("=" * 80)

    print(f"\nOriginal Query: {response.get('query', 'N/A')}")
    print(f"Generated SQL: {response.get('sql', 'N/A')}")
    print(f"Execution Time: {response.get('execution_time_ms', 'N/A')} ms")
    print(f"Records Found: {response.get('record_count', 0)}")

    desc = response.get('description') or {}
    filters: List[str] = desc.get('filters') or []
    if filters:
        print("\nFilters Applied:")
        for f in filters:
            print(f"  - {f}")

    data = response.get('data') or []
    if not data:
        print("\nNo results found.")
        print("\n" + "=" * 80)
        return

    cols = ["id", "posted_date_time", "channel_name", "author", "content"]

    def short(val: Any, width: int) -> str:
        s = "" if val is None else str(val)
        return (s[: width - 3] + "...") if len(s) > width else s

    rows = []
    for rec in data[:10]:
        rows.append([
            short(rec.get("id"), 12),
            short(rec.get("posted_date_time"), 25),
            short(rec.get("channel_name"), 20),
            short(rec.get("author"), 18),
            short(rec.get("content"), 80),
        ])

    try:
        from tabulate import tabulate
        print("\nShowing up to 10 rows:")
        print(tabulate(rows, headers=cols, tablefmt="grid"))
    except Exception:
        print(" | ".join(cols))
        print("-" * 80)
        for r in rows:
            print(" | ".join(map(str, r)))

    if len(data) > 10:
        print(f"\n... and {len(data) - 10} more records")
    print("\n" + "=" * 80)


def interactive_mode(api_base: str, default_limit: int = 10) -> None:
    """Run an interactive REPL for NL→SQL queries."""
    print("\nChanscope NL Query Interface")
    print("=" * 50)
    print("Enter queries in natural language.")
    print("Type 'help' for examples, 'quit' to exit.")
    print("=" * 50)

    while True:
        try:
            print("\nEnter your query (or 'quit' to exit):")
            user_input = input(">>> ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            if user_input.lower() == "help":
                print("\nExamples:")
                print("  - Show me posts from the last hour")
                print("  - Find messages containing bitcoin from yesterday")
                print("  - Get 5 recent posts about AI")
                print("  - Show posts from channel tech in the last 3 days")
                print("  - Find posts by author satoshi containing crypto")
                print("  - What are people saying about market trends?")
                continue
            if user_input.lower() == "clear":
                print("\033[2J\033[H")
                continue
            if not user_input:
                continue

            # extract inline "limit N" if present
            limit = default_limit
            if "limit" in user_input.lower():
                import re as _re
                m = _re.search(r"(\d+)\s*(?:limit|results?|rows?)", user_input.lower())
                if m:
                    limit = int(m.group(1))

            resp = send_query(api_base, user_input, limit)
            display_results(resp)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Replit Postgres via NL→SQL API")
    parser.add_argument("query", nargs="?", help="Natural language query (omit for interactive mode)")
    parser.add_argument("--api", default=_default_api_base(), help="API base URL (auto-detected by default)")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if args.query:
        print(f"\nExecuting query: {args.query}")
        resp = send_query(args.api, args.query, args.limit)
        display_results(resp, as_json=args.json)
        return

    interactive_mode(args.api, args.limit)


if __name__ == "__main__":
    main()

