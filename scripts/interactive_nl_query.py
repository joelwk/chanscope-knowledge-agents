
#!/usr/bin/env python3
"""
Interactive script for testing natural language queries.
Prompts user for input and makes API requests to the nl_query endpoint.
"""

import requests
import json
import sys
from typing import Dict, Any

def get_user_input() -> Dict[str, Any]:
    """Get user input for the natural language query."""
    print("=== Interactive Natural Language Query ===")
    print()
    
    # Get the query
    query = input("Enter your natural language query: ").strip()
    if not query:
        print("Error: Query cannot be empty")
        sys.exit(1)
    
    # Get optional limit
    limit_input = input("Enter result limit (default: 10): ").strip()
    if limit_input:
        try:
            limit = int(limit_input)
            if limit <= 0:
                print("Error: Limit must be positive")
                sys.exit(1)
        except ValueError:
            print("Error: Limit must be a number")
            sys.exit(1)
    else:
        limit = 10
    
    # Ask about saving results
    save_input = input("Save results to disk? (y/N): ").strip().lower()
    save_result = save_input in ['y', 'yes']
    
    return {
        "query": query,
        "limit": limit,
        "save_result": save_result
    }

def make_nl_query_request(params: Dict[str, Any], api_base: str = "http://0.0.0.0") -> None:
    """Make the API request and display results."""
    url = f"{api_base}/api/v1/nl_query"
    
    payload = {
        "query": params["query"],
        "limit": params["limit"],
        "format_for_llm": True
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"\nSending request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\nWaiting for response...")
    
    try:
        response = requests.post(url, json=payload, headers=headers, params={"save_result": params["save_result"]})
        
        if response.status_code == 200:
            result = response.json()
            display_results(result)
        else:
            print(f"Error: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def display_results(result: Dict[str, Any]) -> None:
    """Display the query results in a readable format."""
    print("\n" + "="*60)
    print("QUERY RESULTS")
    print("="*60)
    
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Original Query: {result.get('query', 'N/A')}")
    print(f"Generated SQL: {result.get('sql', 'N/A')}")
    print(f"Record Count: {result.get('record_count', 0)}")
    print(f"Execution Time: {result.get('execution_time_ms', 0)}ms")
    
    # Display query interpretation
    if 'description' in result:
        desc = result['description']
        print(f"\nQuery Interpretation:")
        if 'filters' in desc:
            for filter_desc in desc['filters']:
                print(f"  - {filter_desc}")
    
    # Display data
    if 'data' in result and result['data']:
        print(f"\nData (showing first 5 records):")
        print("-" * 40)
        
        for i, record in enumerate(result['data'][:5]):
            print(f"\nRecord {i+1}:")
            for key, value in record.items():
                if key == 'content' and isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        if len(result['data']) > 5:
            print(f"\n... and {len(result['data']) - 5} more records")
    else:
        print("\nNo data returned")
    
    # Display metadata
    if 'metadata' in result:
        metadata = result['metadata']
        print(f"\nMetadata:")
        if 'saved_files' in metadata:
            print(f"  Saved files: {metadata['saved_files']}")
        if 'providers_used' in metadata:
            print(f"  Providers used: {metadata['providers_used']}")
        print(f"  Timestamp: {metadata.get('timestamp', 'N/A')}")

def main():
    """Main function."""
    try:
        # Get user input
        params = get_user_input()
        
        # Make the API request
        make_nl_query_request(params)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
