import requests
import json
import time
import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:5000"

def test_health() -> bool:
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        result = response.json()
        logger.info(f"Health check response: {result}")
        return response.status_code == 200 and result.get("status") == "healthy"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def test_process_query() -> Dict[str, Any]:
    """Test the process_query endpoint."""
    logger.info("\nTesting process_query endpoint...")
    results = {}
    try:
        # Test with OpenAI
        logger.info("\nTesting with OpenAI model...")
        response = requests.post(
            f"{BASE_URL}/process_query",
            json={
                "query": "What are the latest developments in AI?",
                "process_new": True,
                "batch_size": int(os.getenv('BATCH_SIZE', '100')),
                "model": os.getenv('OPENAI_MODEL'),
                "embedding_provider": "openai",
                "embedding_model": os.getenv('OPENAI_EMBEDDING_MODEL'),
                "summary_provider": "openai"
            }
        )
        result = response.json()
        results['openai'] = result
        logger.info("\nOpenAI model response:")
        if result.get("success"):
            if "results" in result and "chunks" in result["results"]:
                logger.info("\nRelevant chunks:")
                for i, chunk in enumerate(result["results"]["chunks"], 1):
                    logger.info(f"\nChunk {i}:")
                    logger.info(f"Text: {chunk.get('text', 'No text')}")
            if "results" in result and "summary" in result["results"]:
                logger.info(f"\nSummary (OpenAI): {result['results']['summary']}")
        else:
            logger.error(f"OpenAI model error: {result.get('error', 'Unknown error')}")
        
        # Test with Grok
        logger.info("\nTesting with Grok model...")
        response = requests.post(
            f"{BASE_URL}/process_query",
            json={
                "query": "What are the latest developments in AI?",
                "process_new": False,
                "batch_size": int(os.getenv('BATCH_SIZE', '100')),
                "model": os.getenv('GROK_MODEL'),
                "embedding_provider": os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai'),
                "embedding_model": os.getenv('OPENAI_EMBEDDING_MODEL'),
                "summary_provider": "grok"
            }
        )
        result = response.json()
        results['grok'] = result
        logger.info("\nGrok model response:")
        if result.get("success"):
            if "results" in result and "chunks" in result["results"]:
                logger.info("\nRelevant chunks:")
                for i, chunk in enumerate(result["results"]["chunks"], 1):
                    logger.info(f"\nChunk {i}:")
                    logger.info(f"Text: {chunk.get('text', 'No text')}")
            if "results" in result and "summary" in result["results"]:
                logger.info(f"\nSummary (Grok): {result['results']['summary']}")
        else:
            logger.error(f"Grok model error: {result.get('error', 'Unknown error')}")
        
        # Test with Venice
        logger.info("\nTesting with Venice model...")
        response = requests.post(
            f"{BASE_URL}/process_query",
            json={
                "query": "What are the latest developments in AI?",
                "process_new": False,
                "batch_size": int(os.getenv('BATCH_SIZE', '100')),
                "model": os.getenv('VENICE_MODEL'),
                "embedding_provider": os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai'),
                "embedding_model": os.getenv('OPENAI_EMBEDDING_MODEL'),
                "summary_provider": "venice"
            }
        )
        result = response.json()
        results['venice'] = result
        logger.info("\nVenice model response:")
        if result.get("success"):
            if "results" in result and "chunks" in result["results"]:
                logger.info("\nRelevant chunks:")
                for i, chunk in enumerate(result["results"]["chunks"], 1):
                    logger.info(f"\nChunk {i}:")
                    logger.info(f"Text: {chunk.get('text', 'No text')}")
            if "results" in result and "summary" in result["results"]:
                logger.info(f"\nSummary (Venice): {result['results']['summary']}")
        else:
            logger.error(f"Venice model error: {result.get('error', 'Unknown error')}")
        
        assert response.status_code == 200, "Process query failed"
        logger.info("Process query tests passed for all models")
        return results
    except Exception as e:
        logger.error(f"Process query test failed: {e}")
        raise

def test_batch_process() -> Dict[str, Any]:
    """Test the batch_process endpoint."""
    logger.info("\nTesting batch_process endpoint...")
    try:
        # Test with multiple queries and different model providers
        response = requests.post(
            f"{BASE_URL}/batch_process",
            json={
                "queries": [
                    "What are the latest developments in AI?",
                    "How is AI impacting healthcare?",
                    "What are the ethical concerns around AI?"
                ],
                "process_new": False,
                "batch_size": int(os.getenv('BATCH_SIZE', '100')),
                "embedding_provider": os.getenv('DEFAULT_EMBEDDING_PROVIDER', 'openai'),
                "embedding_model": os.getenv('OPENAI_EMBEDDING_MODEL'),
                "summary_provider": "venice"  # Using Venice for summaries
            }
        )
        result = response.json()
        logger.info("\nBatch process response:")
        if result.get("success"):
            for i, query_result in enumerate(result["results"], 1):
                logger.info(f"\nQuery {i}: {query_result['query']}")
                if "chunks" in query_result:
                    logger.info("\nRelevant chunks:")
                    for j, chunk in enumerate(query_result["chunks"], 1):
                        logger.info(f"\nChunk {j}:")
                        logger.info(f"Text: {chunk.get('text', 'No text')}")
                if "summary" in query_result:
                    logger.info(f"\nSummary: {query_result['summary']}")
        else:
            logger.error(f"Batch process error: {result.get('error', 'Unknown error')}")
        
        assert response.status_code == 200, "Batch process failed"
        logger.info("Batch process test passed")
        return result
    except Exception as e:
        logger.error(f"Batch process test failed: {e}")
        raise

def wait_for_service(timeout: int = 30) -> bool:
    """Wait for the service to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if test_health():
                logger.info("Service is ready!")
                return True
            logger.info("Service not ready yet, waiting...")
        except:
            pass
        time.sleep(2)
    return False

def main():
    """Run all tests."""
    logger.info("Starting endpoint tests...")
    
    # Wait for service to be ready
    if not wait_for_service():
        logger.error("Service did not become available in time")
        return
    
    # Test health endpoint
    if not test_health():
        logger.error("Health check failed")
        return
    
    # Test process_query endpoint
    logger.info("\nTesting process_query endpoint...")
    result = test_process_query()
    if "error" in result:
        logger.error("Process query test failed")
    
    # Test batch_process endpoint
    logger.info("\nTesting batch_process endpoint...")
    result = test_batch_process()
    if "error" in result:
        logger.error("Batch process test failed")
    
    logger.info("All tests completed!")

if __name__ == "__main__":
    main() 