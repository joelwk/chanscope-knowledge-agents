"""Test script for LLM chunking optimizations."""
import os
import sys
import asyncio
import json
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge_agents.model_ops import KnowledgeAgent, ModelProvider

# Sample test content
TEST_CONTENT_SHORT = "This is a short test content for chunking."

TEST_CONTENT_MEDIUM = """
The cryptocurrency market has seen significant volatility in recent weeks. 
Bitcoin prices fluctuated between $45,000 and $52,000, driven by institutional 
investment news and regulatory developments. Ethereum showed strong performance 
with the upcoming network upgrades. Many analysts believe the current market 
conditions present both opportunities and risks for investors. The DeFi sector 
continues to grow, with total value locked exceeding $100 billion across 
various protocols. NFT trading volumes have decreased from their peak but 
remain substantial in certain collections.
"""

TEST_CONTENT_LONG = """
The rapid advancement of artificial intelligence has transformed numerous industries 
in unprecedented ways. Machine learning algorithms now power recommendation systems, 
autonomous vehicles, medical diagnostics, and financial trading platforms. Natural 
language processing has evolved to enable sophisticated chatbots and translation 
services that can understand context and nuance.

In the healthcare sector, AI assists doctors in diagnosing diseases, predicting 
patient outcomes, and personalizing treatment plans. Computer vision technology 
can detect cancerous cells in medical images with accuracy that sometimes surpasses 
human specialists. Drug discovery processes have been accelerated through AI 
models that can predict molecular interactions and identify promising compounds.

The ethical implications of AI deployment remain a critical concern. Issues such 
as algorithmic bias, privacy protection, and the potential for job displacement 
require careful consideration. Regulatory frameworks are struggling to keep pace 
with technological advancement, leading to calls for more comprehensive governance 
structures.

Looking ahead, the integration of AI with other emerging technologies like quantum 
computing and biotechnology promises even more transformative changes. Researchers 
are working on artificial general intelligence (AGI) that could match or exceed 
human cognitive abilities across all domains. While the timeline for AGI remains 
uncertain, its potential impact on society would be profound.

Educational institutions are adapting their curricula to prepare students for an 
AI-driven future. Skills in data science, machine learning, and AI ethics are 
becoming increasingly valuable. Organizations are investing heavily in AI talent 
and infrastructure to maintain competitive advantages in their respective markets.
"""

async def test_deterministic_chunking():
    """Test deterministic text chunking."""
    print("\n=== Testing Deterministic Chunking ===")
    
    # Enable deterministic chunking
    os.environ['USE_DETERMINISTIC_CHUNKING'] = 'true'
    os.environ['CHUNK_SIZE'] = '500'
    os.environ['CHUNK_OVERLAP'] = '100'
    
    agent = KnowledgeAgent()
    
    # Test with different content lengths
    for name, content in [("SHORT", TEST_CONTENT_SHORT), 
                         ("MEDIUM", TEST_CONTENT_MEDIUM), 
                         ("LONG", TEST_CONTENT_LONG)]:
        print(f"\nTesting {name} content ({len(content)} chars):")
        
        start_time = time.time()
        result = await agent.generate_chunks(content)
        elapsed = time.time() - start_time
        
        print(f"  - Thread ID: {result.get('thread_id', 'N/A')}")
        print(f"  - Number of chunks: {len(result.get('chunks', []))}")
        print(f"  - Summary length: {len(result.get('summary', ''))}")
        print(f"  - Processing time: {elapsed:.3f}s")
        
        # Verify chunk sizes
        chunks = result.get('chunks', [])
        if chunks:
            chunk_lengths = [len(chunk) for chunk in chunks]
            print(f"  - Chunk lengths: {chunk_lengths}")

async def test_llm_chunking_with_retry():
    """Test LLM-based chunking with retry logic."""
    print("\n=== Testing LLM Chunking with Retry ===")
    
    # Disable deterministic chunking
    os.environ['USE_DETERMINISTIC_CHUNKING'] = 'false'
    os.environ['CHUNK_MAX_RETRIES'] = '3'
    
    agent = KnowledgeAgent()
    
    print("\nTesting MEDIUM content with LLM:")
    start_time = time.time()
    
    try:
        result = await agent.generate_chunks(TEST_CONTENT_MEDIUM)
        elapsed = time.time() - start_time
        
        print(f"  - Success: {isinstance(result, dict)}")
        print(f"  - Has required keys: {all(k in result for k in ['thread_id', 'chunks', 'summary'])}")
        print(f"  - Processing time: {elapsed:.3f}s")
        
        # Try to pretty print the result
        if isinstance(result, dict):
            print("\n  Result structure:")
            print(f"    - thread_id: {result.get('thread_id', 'N/A')}")
            print(f"    - chunks: {len(result.get('chunks', []))} items")
            print(f"    - summary: {result.get('summary', 'N/A')[:100]}...")
            
    except Exception as e:
        print(f"  - Error: {str(e)}")
        elapsed = time.time() - start_time
        print(f"  - Failed after: {elapsed:.3f}s")

async def test_batch_chunking():
    """Test batch chunking with parallel processing."""
    print("\n=== Testing Batch Chunking ===")
    
    # Enable deterministic chunking for consistent results
    os.environ['USE_DETERMINISTIC_CHUNKING'] = 'true'
    os.environ['CHUNK_SIZE'] = '300'
    
    agent = KnowledgeAgent()
    
    # Create multiple test contents
    test_contents = [
        f"Test content {i}: " + TEST_CONTENT_SHORT 
        for i in range(10)
    ]
    
    print(f"\nProcessing {len(test_contents)} items in batch:")
    
    start_time = time.time()
    results = await agent.generate_chunks_batch(test_contents, batch_size=3)
    elapsed = time.time() - start_time
    
    successful = sum(1 for r in results if r is not None)
    print(f"  - Successful: {successful}/{len(test_contents)}")
    print(f"  - Total processing time: {elapsed:.3f}s")
    print(f"  - Average time per item: {elapsed/len(test_contents):.3f}s")

async def test_fallback_behavior():
    """Test fallback from LLM to deterministic chunking."""
    print("\n=== Testing Fallback Behavior ===")
    
    # Configure for quick fallback
    os.environ['USE_DETERMINISTIC_CHUNKING'] = 'false'
    os.environ['CHUNK_MAX_RETRIES'] = '1'  # Fail fast
    
    # Temporarily break the API key to force fallback
    original_key = os.environ.get('OPENAI_API_KEY', '')
    os.environ['OPENAI_API_KEY'] = 'invalid_key_to_force_error'
    
    agent = KnowledgeAgent()
    
    print("\nTesting fallback with invalid API key:")
    start_time = time.time()
    
    try:
        result = await agent.generate_chunks(TEST_CONTENT_SHORT)
        elapsed = time.time() - start_time
        
        # Should fall back to deterministic chunking
        print(f"  - Fallback successful: {isinstance(result, dict)}")
        print(f"  - Processing time: {elapsed:.3f}s")
        print(f"  - Result has chunks: {len(result.get('chunks', []))} items")
        
    except Exception as e:
        print(f"  - Unexpected error: {str(e)}")
    
    finally:
        # Restore original API key
        os.environ['OPENAI_API_KEY'] = original_key

async def main():
    """Run all tests."""
    print("Starting LLM Chunking Optimization Tests")
    print("=" * 50)
    
    # Test deterministic chunking
    await test_deterministic_chunking()
    
    # Test LLM chunking with retry (skip if no API key)
    if os.environ.get('OPENAI_API_KEY'):
        await test_llm_chunking_with_retry()
    else:
        print("\n=== Skipping LLM tests (no OPENAI_API_KEY) ===")
    
    # Test batch processing
    await test_batch_chunking()
    
    # Test fallback behavior
    await test_fallback_behavior()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
