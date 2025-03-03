"""Simple script to test the cache health endpoint directly."""
from fastapi.testclient import TestClient
import json

# Import the app instance
from api.app import app

# Create test client
client = TestClient(app)

# Define API base path
API_BASE_PATH = "/api/v1"

def test_cache_health():
    """Test the cache health endpoint directly."""
    try:
        # Try the endpoint
        response = client.get(f"{API_BASE_PATH}/health/cache")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response data:")
            print(json.dumps(data, indent=2))
            
            # Check expected fields
            if "metrics" in data:
                print("\nMetrics found in response")
                metrics = data["metrics"]
                print(f"- hit_ratio: {metrics.get('hit_ratio', 'N/A')}")
                print(f"- hits: {metrics.get('hits', 'N/A')}")
                print(f"- misses: {metrics.get('misses', 'N/A')}")
                print(f"- errors: {metrics.get('errors', 'N/A')}")
            else:
                print("\nNo metrics found in response")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cache_health() 