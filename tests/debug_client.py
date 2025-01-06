import json
import requests
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DebugClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        """Test the API health endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return None

    def run_single_query(self, query_data):
        """Run a single query through the API."""
        try:
            response = self.session.post(
                f"{self.base_url}/process_query",
                json=query_data
            )
            return response.json()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None

    def run_batch_query(self, batch_data):
        """Run a batch of queries through the API."""
        try:
            response = self.session.post(
                f"{self.base_url}/batch_process",
                json=batch_data
            )
            return response.json()
        except Exception as e:
            logger.error(f"Batch query failed: {e}")
            return None

def load_test_queries(file_path):
    """Load test queries from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load test queries: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Debug client for Knowledge Agent API')
    parser.add_argument('--url', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--query-type', choices=['single', 'batch', 'temporal', 'board', 'debug'],
                      default='debug', help='Type of query to run')
    args = parser.parse_args()

    # Initialize client
    client = DebugClient(args.url)

    # Check API health
    logger.info("Checking API health...")
    health = client.test_health()
    if not health:
        logger.error("API is not healthy. Exiting.")
        return

    logger.info(f"API health check response: {health}")

    # Load test queries
    queries = load_test_queries(Path(__file__).parent / 'test_queries.json')
    if not queries:
        return

    # Run queries based on type
    if args.query_type == 'single':
        for query in queries['single_queries']:
            logger.info(f"Running single query: {query['query']}")
            result = client.run_single_query(query)
            logger.info(f"Result: {json.dumps(result, indent=2)}")
    
    elif args.query_type == 'batch':
        logger.info("Running batch queries")
        result = client.run_batch_query(queries['batch_queries'])
        logger.info(f"Result: {json.dumps(result, indent=2)}")
    
    elif args.query_type == 'temporal':
        logger.info("Running temporal analysis query")
        result = client.run_single_query(queries['temporal_analysis'])
        logger.info(f"Result: {json.dumps(result, indent=2)}")
    
    elif args.query_type == 'board':
        for board, query in queries['board_specific'].items():
            logger.info(f"Running query for board /{board}/")
            result = client.run_single_query(query)
            logger.info(f"Result: {json.dumps(result, indent=2)}")
    
    elif args.query_type == 'debug':
        logger.info("Running debug queries")
        for name, query in queries['debug_queries'].items():
            logger.info(f"Running debug query: {name}")
            result = client.run_single_query(query)
            logger.info(f"Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main() 