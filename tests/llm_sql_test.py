#!/usr/bin/env python
"""
Simplified test module for LLM SQL Generator functionality.

This module provides basic tests for the LLM SQL Generator without complex mocking
or advanced testing patterns.

Usage:
- Run tests: python -m tests.llm_sql_test
"""

import unittest
import sys
import os
import asyncio
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Any

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_agents.llm_sql_generator import LLMSQLGenerator
from knowledge_agents.model_ops import KnowledgeAgent, ModelProvider
from knowledge_agents.embedding_ops import get_agent

# -----------------------------------------------------------------------------
# TEST DATA
# -----------------------------------------------------------------------------

# Sample test queries
TEST_QUERIES = {
    "time_based": [
        "Give me threads from the last hour",
        "Show posts from yesterday containing crypto",
        "Find messages from the last 3 days by author john"
    ],
    "content_based": [
        "Find messages containing bitcoin",
        "Show posts about AI and machine learning"
    ],
    "complex": [
        "Get threads from board tech about AI from this week",
        "Find posts by author satoshi about bitcoin from yesterday"
    ],
    "limit_based": [
        "Show me 5 recent posts",
        "Get 10 random rows",
        "Find 20 messages containing crypto"
    ]
}

# -----------------------------------------------------------------------------
# BASIC TESTS
# -----------------------------------------------------------------------------

class BasicLLMSQLTests(unittest.TestCase):
    """
    Basic functionality tests for LLM SQL Generator.
    
    These tests focus on the core functionality without complex mocking
    or extensive test infrastructure.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        print("Setting up LLM SQL Generator tests...")
        # Create real agent and generator - these tests use the actual implementation
        cls.agent = None
        cls.generator = None
        
        try:
            # Create the agent and generator synchronously
            cls.agent = asyncio.run(get_agent())
            cls.generator = LLMSQLGenerator(cls.agent)
            print("LLM SQL Generator initialized successfully")
        except Exception as e:
            print(f"Error initializing test environment: {e}")
            raise
    
    def test_sql_templates(self):
        """Test that SQL templates are correctly initialized."""
        templates = self.generator._templates
        
        # Check that essential template categories exist
        self.assertIn("time_based", templates)
        self.assertIn("content_based", templates)
        self.assertIn("author_based", templates)
        self.assertIn("default", templates)
        
        # Check time-based templates
        time_templates = templates["time_based"]
        self.assertIn("last_hour", time_templates)
        self.assertIn("today", time_templates)
        
        # Check content-based template
        content_template = templates["content_based"]
        self.assertIn("pattern", content_template)
        self.assertIn("sql", content_template)
        self.assertIn("param_extractor", content_template)
    
    def test_template_based_generation(self):
        """Test the unified template-based SQL generation."""
        # Test time-based template
        nl_query = "Give me threads from the last hour"
        sql, params, generators = self._run_sync(self.generator._generate_sql_from_templates(nl_query))
        
        self.assertIsNotNone(sql, "Should generate valid SQL")
        self.assertIn("posted_date_time >= %s", sql)
        self.assertEqual(len(params), 1, "Should have one parameter")
        
        # Verify parameter is approximately an hour ago
        now = datetime.now(pytz.UTC)
        time_diff = now - params[0]
        self.assertTrue(timedelta(minutes=55) < time_diff < timedelta(minutes=65),
                        "Parameter should be about an hour ago")
        
        # Test content-based template
        nl_query = "Find messages containing bitcoin"
        sql, params, generators = self._run_sync(self.generator._generate_sql_from_templates(nl_query))
        
        self.assertIsNotNone(sql, "Should generate valid SQL")
        self.assertIn("content ILIKE %s", sql)
        self.assertEqual(params[0], "%bitcoin%", "Should have correct content parameter")
        
        # Test combined query with limit
        nl_query = "Find 5 messages containing bitcoin by author satoshi"
        sql, params, generators = self._run_sync(self.generator._generate_sql_from_templates(nl_query))
        
        self.assertIsNotNone(sql, "Should generate valid SQL")
        self.assertIn("content ILIKE %s", sql)
        self.assertIn("author ILIKE %s", sql)
        self.assertIn("LIMIT 5", sql)
        self.assertEqual(params[0], "%bitcoin%", "Should have correct content parameter")
        self.assertEqual(params[1], "%satoshi%", "Should have correct author parameter")
    
    def test_limit_extraction(self):
        """Test extracting limit parameter from queries."""
        # Test with explicit limit
        limit = self.generator._extract_limit("Show me 15 posts")
        self.assertEqual(limit, 15, "Should extract limit correctly")
        
        # Test with no limit specified
        limit = self.generator._extract_limit("Show me recent posts", default=50)
        self.assertEqual(limit, 50, "Should use default limit")
        
        # Test with limit in different position
        limit = self.generator._extract_limit("Get posts limit 25")
        self.assertEqual(limit, 25, "Should extract limit correctly")
    
    def test_query_description(self):
        """Test generating natural language query descriptions."""
        # Test time-based description
        description = self.generator.get_query_description("Show posts from the last 5 hours")
        self.assertEqual(description["time_filter"], "Last 5 hours")
        self.assertIn("Time: Last 5 hours", description["filters"])
        
        # Test content description
        description = self.generator.get_query_description("Find messages containing bitcoin")
        self.assertEqual(description["content_filter"], "bitcoin")
        self.assertIn("Content: Contains 'bitcoin'", description["filters"])
        
        # Test combined description
        description = self.generator.get_query_description("Find posts by author satoshi about bitcoin")
        self.assertEqual(description["content_filter"], "bitcoin")
        self.assertEqual(description["author_filter"], "satoshi")
        self.assertIn("Content: Contains 'bitcoin'", description["filters"])
        self.assertIn("Author: 'satoshi'", description["filters"])
        
        # Test limit description
        description = self.generator.get_query_description("Show me 10 recent posts")
        self.assertEqual(description["limit"], 10)
        self.assertIn("Limit: 10 rows", description["filters"])
    
    def test_simple_validation(self):
        """Test the simple validation fallback for SQL queries."""
        # Valid SELECT query
        sql = "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC"
        result = self.generator._simple_validation_fallback(sql)
        self.assertTrue(result["is_safe"], "Valid SQL should be safe")
        self.assertEqual(result["reason"], "Safe")
        
        # Invalid non-SELECT query
        sql = "UPDATE complete_data SET author = 'test'"
        result = self.generator._simple_validation_fallback(sql)
        self.assertFalse(result["is_safe"], "Non-SELECT should be unsafe")
        self.assertIn("Only SELECT statements are allowed", result["reason"])
        
        # Query with no parameters but with LIMIT is valid
        sql = "SELECT * FROM complete_data ORDER BY posted_date_time DESC LIMIT 10"
        result = self.generator._simple_validation_fallback(sql)
        self.assertTrue(result["is_safe"], "Query with LIMIT but no parameters should be safe")
        
        # Query with incorrect table
        sql = "SELECT * FROM other_table WHERE id = %s"
        result = self.generator._simple_validation_fallback(sql)
        self.assertFalse(result["is_safe"], "Query with wrong table should be unsafe")
    
    def test_parameter_extraction(self):
        """Test extracting parameters from natural language queries."""
        # Time-based parameter
        nl_query = "Show posts from the last 3 hours"
        sql = "SELECT * FROM complete_data WHERE posted_date_time >= %s"
        
        params, _ = self._run_sync(self.generator._extract_parameters(
            nl_query, sql, ["Time parameter"]))
        
        self.assertEqual(len(params), 1, "Should have one parameter")
        self.assertTrue(isinstance(params[0], datetime), "Parameter should be a datetime")
        
        now = datetime.now(pytz.UTC)
        time_diff = now - params[0]
        self.assertTrue(timedelta(hours=2.9) < time_diff < timedelta(hours=3.1),
                       "Parameter should be about 3 hours ago")
        
        # Content parameter
        nl_query = "Find messages containing bitcoin"
        sql = "SELECT * FROM complete_data WHERE content ILIKE %s"
        
        params, _ = self._run_sync(self.generator._extract_parameters(
            nl_query, sql, ["Content parameter"]))
        
        self.assertEqual(len(params), 1, "Should have one parameter")
        self.assertEqual(params[0], "%bitcoin%", "Should have correct content parameter")
        
        # Multiple parameters
        nl_query = "Find posts by author satoshi about bitcoin"
        sql = "SELECT * FROM complete_data WHERE content ILIKE %s AND author ILIKE %s"
        
        params, _ = self._run_sync(self.generator._extract_parameters(
            nl_query, sql, ["Content parameter", "Author parameter"]))
        
        self.assertEqual(len(params), 2, "Should have two parameters")
        self.assertEqual(params[0], "%bitcoin%", "Should have correct content parameter")
        self.assertEqual(params[1], "%satoshi%", "Should have correct author parameter")
    
    def test_end_to_end_simple_query(self):
        """Test a simple end-to-end query."""
        # Simple time-based query that should use template matching
        nl_query = "Show posts from the last hour"
        
        sql, params = self._run_sync(self.generator.generate_sql(nl_query))
        
        # Verify the generated SQL
        self.assertIsNotNone(sql, "Should generate valid SQL")
        self.assertTrue(sql.upper().startswith("SELECT"), "Should be a SELECT statement")
        self.assertIn("complete_data", sql, "Should query the complete_data table")
        self.assertIn("posted_date_time >= %s", sql, "Should have time-based filter")
        self.assertEqual(len(params), 1, "Should have one parameter")
        
        # Parameter should be about an hour ago
        now = datetime.now(pytz.UTC)
        time_diff = now - params[0]
        self.assertTrue(timedelta(minutes=55) < time_diff < timedelta(minutes=65),
                       "Parameter should be about an hour ago")
        
        # Query with limit
        nl_query = "Show me 5 recent posts"
        
        sql, params = self._run_sync(self.generator.generate_sql(nl_query))
        
        # Verify the generated SQL has a LIMIT clause
        self.assertIn("LIMIT 5", sql, "Should include the specified LIMIT")
    
    def test_random_queries(self):
        """Test that random queries are properly handled, including with time filters."""
        # Simple random query
        nl_query = "Give me 10 random rows"
        sql, params, generators = self._run_sync(self.generator._generate_sql_from_templates(nl_query))
        
        self.assertIsNotNone(sql, "Should generate valid SQL")
        self.assertIn("ORDER BY RANDOM()", sql, "Should use RANDOM() ordering")
        self.assertIn("LIMIT 10", sql, "Should include default limit")
        self.assertEqual(len(params), 0, "Should have no parameters for simple random query")
        
        # Random query with time filter
        nl_query = "Find 5 random rows from the last week"
        sql, params, generators = self._run_sync(self.generator._generate_sql_from_templates(nl_query))
        
        self.assertIsNotNone(sql, "Should generate valid SQL")
        self.assertIn("WHERE posted_date_time >= %s", sql, "Should include time filter")
        self.assertIn("ORDER BY RANDOM()", sql, "Should use RANDOM() ordering")
        self.assertIn("LIMIT 5", sql, "Should include specified limit")
        self.assertEqual(len(params), 1, "Should have one parameter for time filter")
        self.assertTrue(isinstance(params[0], datetime), "Parameter should be a datetime")
        
        # Random query with content filter
        nl_query = "Find 5 random rows containing bitcoin"
        sql, params, generators = self._run_sync(self.generator._generate_sql_from_templates(nl_query))
        
        self.assertIsNotNone(sql, "Should generate valid SQL")
        self.assertIn("WHERE content ILIKE %s", sql, "Should include content filter")
        self.assertIn("ORDER BY RANDOM()", sql, "Should use RANDOM() ordering")
        self.assertEqual(len(params), 1, "Should have one parameter for content filter")
        self.assertEqual(params[0], "%bitcoin%", "Parameter should have correct content value")
        
        # Random query with both time and content filters
        nl_query = "Find 10 random rows from the last week containing bitcoin"
        sql, params, generators = self._run_sync(self.generator._generate_sql_from_templates(nl_query))
        
        self.assertIsNotNone(sql, "Should generate valid SQL")
        self.assertIn("WHERE posted_date_time >= %s AND content ILIKE %s", sql, "Should include both time and content filters")
        self.assertIn("ORDER BY RANDOM()", sql, "Should use RANDOM() ordering")
        self.assertEqual(len(params), 2, "Should have two parameters")
        self.assertTrue(isinstance(params[0], datetime), "First parameter should be a datetime")
        self.assertEqual(params[1], "%bitcoin%", "Second parameter should have correct content value")
        
        # Verify actual generated SQL
        nl_query = "Find 10 random rows from the last week"
        sql, params = self._run_sync(self.generator.generate_sql(nl_query))
        
        self.assertIn("ORDER BY RANDOM()", sql, "Should use RANDOM() ordering in final SQL")
        self.assertEqual(len(params), 1, "Should have one parameter for time filter")
        self.assertTrue(isinstance(params[0], datetime), "Parameter should be a datetime")
    
    def _run_sync(self, coroutine):
        """Helper method to run a coroutine synchronously."""
        import asyncio
        
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the coroutine and get the result
            return loop.run_until_complete(coroutine)
        finally:
            # Clean up the event loop
            loop.close()
            asyncio.set_event_loop(None)

# -----------------------------------------------------------------------------
# MANUAL TESTING
# -----------------------------------------------------------------------------

def run_manual_test(query):
    """Run a manual test with a specific query."""
    import asyncio
    
    async def _async_test():
        print(f"Testing query: {query}")
        
        # Initialize agent and generator
        agent = await get_agent()
        generator = LLMSQLGenerator(agent)
        
        # Get query description
        description = generator.get_query_description(query)
        print("\nQuery interpretation:")
        for filter_desc in description["filters"]:
            print(f"- {filter_desc}")
        
        # Generate SQL
        try:
            print("\nGenerating SQL...")
            start_time = datetime.now()
            sql, params = await generator.generate_sql(query)
            end_time = datetime.now()
            
            print(f"\nGenerated SQL: {sql}")
            print(f"With parameters: {params}")
            print(f"Generation time: {(end_time - start_time).total_seconds():.2f} seconds")
            
            return True
        except Exception as e:
            print(f"\nError generating SQL: {e}")
            return False
    
    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(_async_test())
    finally:
        loop.close()
        asyncio.set_event_loop(None)

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM SQL Generator")
    parser.add_argument("--query", type=str, help="Run with a specific query")
    parser.add_argument("--unittest", action="store_true", help="Run unit tests")
    
    args = parser.parse_args()
    
    if args.query:
        # Run manual test with specific query
        success = run_manual_test(args.query)
        sys.exit(0 if success else 1)
    elif args.unittest:
        # Run unit tests
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        # Run a default set of manual tests
        print("=== Running default tests ===")
        sample_queries = [
            "Show posts from the last hour",
            "Find messages containing bitcoin",
            "Show me 5 recent posts",
            "Find 10 random rows from last week",
            "Show posts by author john containing crypto"
        ]
        
        for query in sample_queries:
            print("\n" + "=" * 50)
            success = run_manual_test(query)
            print("=" * 50)
        
        print("\n=== Testing complete ===")
