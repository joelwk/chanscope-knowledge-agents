#!/usr/bin/env python
"""
Test script for the Venice character slug functionality.

This script tests the get_venice_character_slug function and verifies that
it correctly retrieves the character slug based on the priority order.
"""

import os
import sys
import logging
from typing import Optional

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agents.utils import get_venice_character_slug
from knowledge_agents.model_ops import KnowledgeAgent, ModelProvider
from config.logging_config import get_logger

logger = get_logger(__name__)

async def test_character_slug():
    """Test the get_venice_character_slug function with different scenarios."""
    # Test default case
    default_slug = get_venice_character_slug()
    print(f"Default character slug: {default_slug}")
    
    # Test with explicit slug
    explicit_slug = get_venice_character_slug("custom-character")
    print(f"Explicit character slug: {explicit_slug}")
    assert explicit_slug == "custom-character", "Explicit slug should take highest priority"
    
    # Test with environment variable
    original_env = os.environ.get("VENICE_CHARACTER_SLUG")
    try:
        os.environ["VENICE_CHARACTER_SLUG"] = "env-character"
        env_slug = get_venice_character_slug()
        print(f"Environment character slug: {env_slug}")
        assert env_slug == "env-character", "Environment slug should take priority when no explicit slug is provided"
    finally:
        # Restore original environment variable
        if original_env:
            os.environ["VENICE_CHARACTER_SLUG"] = original_env
        else:
            os.environ.pop("VENICE_CHARACTER_SLUG", None)
    
    print("Character slug tests passed!")

async def test_llm_sql_generator():
    """Test the LLMSQLGenerator with the Venice character slug."""
    try:
        from knowledge_agents.llm_sql_generator import LLMSQLGenerator
        
        # Initialize the agent
        agent = await KnowledgeAgent.create()
        
        # Create SQL generator
        sql_generator = LLMSQLGenerator(agent)
        
        # Test a simple query
        nl_query = "Give me threads from the last hour"
        print(f"Testing query: {nl_query}")
        
        # Check that the class has the static provider configurations
        print(f"PROVIDER_ENHANCER: {sql_generator.PROVIDER_ENHANCER}")
        print(f"PROVIDER_GENERATOR: {sql_generator.PROVIDER_GENERATOR}")
        
        # Try the query with the default Venice character
        sql_query, params = await sql_generator.generate_sql(
            nl_query=nl_query,
            use_hybrid_approach=True
        )
        
        print(f"Generated SQL: {sql_query}")
        print(f"With parameters: {params}")
        
        print("LLMSQLGenerator test completed!")
    except Exception as e:
        print(f"Error testing LLMSQLGenerator: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run the test script."""
    print("===== Venice Character Slug Test =====")
    
    await test_character_slug()
    print("\n")
    await test_llm_sql_generator()
    
    print("===== Testing completed =====")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 