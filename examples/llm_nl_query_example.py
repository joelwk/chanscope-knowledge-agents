#!/usr/bin/env python
"""
Example script demonstrating the LLM-based natural language query API endpoint.

This script sends various natural language queries to the API and displays the results.
"""

import requests
import json
from datetime import datetime
from tabulate import tabulate
import argparse
import time

def send_nl_query(api_base, query, limit=10, provider=None):
    """
    Send a natural language query to the API.
    
    Args:
        api_base: Base URL of the API
        query: Natural language query string
        limit: Maximum number of results to return
        provider: Optional model provider to use
        
    Returns:
        API response JSON
    """
    url = f"{api_base}/api/v1/nl_query"
    
    payload = {
        "query": query,
        "limit": limit
    }
    
    if provider:
        payload["provider"] = provider
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"\nSending query: '{query}'")
    start_time = time.time()
    
    response = requests.post(url, json=payload, headers=headers)
    
    request_time = time.time() - start_time
    print(f"Request completed in {request_time:.2f}s")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def display_results(response):
    """
    Display query results in a readable format.
    
    Args:
        response: API response JSON
    """
    if not response:
        return
    
    # Display query information
    print("\n==== QUERY INFO ====")
    print(f"Original query: {response['query']}")
    print(f"Generated SQL: {response['sql']}")
    
    # Show timing information
    print("\n==== TIMING INFO ====")
    print(f"Total execution time: {response['execution_time_ms']} ms")
    if "metadata" in response and "processing_time_ms" in response["metadata"]:
        print(f"Processing time: {response['metadata']['processing_time_ms']} ms")
    
    # Display how the query was interpreted
    print("\n==== QUERY INTERPRETATION ====")
    for filter_desc in response['description']['filters']:
        print(f"- {filter_desc}")
    
    # Display results in a table
    if response['data']:
        print(f"\n==== RESULTS ({response['record_count']} records) ====")
        
        # Extract a subset of columns for display
        columns = ["thread_id", "posted_date_time", "author", "channel_name"]
        content_preview_len = 50
        
        # Create a list of rows with selected columns
        rows = []
        for record in response['data']:
            row = [record.get(col, "") for col in columns]
            
            # Add content preview
            content = record.get("content", "")
            if content:
                if len(content) > content_preview_len:
                    content = content[:content_preview_len] + "..."
                row.append(content)
            else:
                row.append("")
            
            rows.append(row)
        
        # Create headers for tabulate
        headers = columns + ["content_preview"]
        
        # Display the table
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("\nNo results found.")
    
    # Show metadata
    if "metadata" in response:
        print("\n==== METADATA ====")
        print(f"SQL generation method: {response['metadata'].get('sql_generation_method', 'unknown')}")
        print(f"Model provider: {response['metadata'].get('provider', 'default')}")
        print(f"Timestamp: {response['metadata'].get('timestamp', 'unknown')}")

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Example script for LLM-based natural language database queries")
    parser.add_argument("--api", default="http://localhost", help="API base URL")
    parser.add_argument("--query", help="Natural language query to execute")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    parser.add_argument("--provider", help="Model provider to use (openai, grok, venice)")
    parser.add_argument("--demo", action="store_true", help="Run a demonstration with predefined queries")
    
    args = parser.parse_args()
    
    # Either run demo mode or a single query
    if args.demo:
        print("=== LLM-Based Natural Language Query Demo ===")
        demo_queries = [
            "Give me threads from the last hour",
            "Show posts from yesterday containing crypto",
            "Find messages from the last 3 days by author john",
            "Get threads from board tech about AI from this week",
            "Show messages containing machine learning from this month",
            "What are people saying about market trends this week?",
            "Find recent discussions about the price of Bitcoin"
        ]
        
        for query in demo_queries:
            print(f"\n\n=========== EXECUTING QUERY: '{query}' ===========")
            response = send_nl_query(args.api, query, args.limit, args.provider)
            display_results(response)
            
        print("\n=== Demo Complete ===")
    elif args.query:
        response = send_nl_query(args.api, args.query, args.limit, args.provider)
        display_results(response)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 