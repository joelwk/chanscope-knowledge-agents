"""
LLM-based SQL Generator for natural language queries.

This module provides a sophisticated approach to converting natural language queries
into parameterized SQL using a unified template-based approach with LLM fallback.
"""

import re
import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Pattern, Callable
from datetime import datetime, timedelta
import pytz
try:
    import sqlparse
except Exception:  # pragma: no cover - optional dependency
    sqlparse = None

from knowledge_agents.model_ops import (
    KnowledgeAgent,
    ModelProvider,
    ModelOperation
)

from knowledge_agents.utils import get_venice_character_slug

# Configure logging
logger = logging.getLogger(__name__)

class NLQueryParsingError(Exception):
    """Exception raised when a natural language query cannot be parsed."""
    pass

class LLMSQLGenerator:
    """
    Generate SQL from natural language using a hybrid approach.
    
    This class converts natural language queries to SQL by:
    1. First attempting to match common query patterns using templates
    2. Falling back to a three-stage LLM pipeline:
       a. Enhancer LLM to refine the query into structured instructions
       b. Generator LLM to convert enhanced instructions to SQL
       c. Validator LLM to ensure security and correctness
    """
    
    # Static provider configurations for the pipeline stages
    PROVIDER_ENHANCER = ModelProvider.OPENAI  # For enhancing instructions
    PROVIDER_GENERATOR = ModelProvider.VENICE  # For SQL generation and validation
    
    # Character slug configurations for each pipeline stage
    #CHARACTER_ENHANCER = "the-architect-of-precision-the-architect"  # For enhancing instructions
    CHARACTER_ENHANCER = "the-architect-of-precision-the-architect"  # For enhancing instructions
    CHARACTER_GENERATOR = "sqlman"  # For SQL generation
    CHARACTER_VALIDATOR = "sqlman"  # For SQL validation

    # Table schema information (used in prompts)
    SCHEMA_DEFINITION = """
    Table: complete_data
    Columns:
    - id (SERIAL PRIMARY KEY): Unique identifier for each record
    - thread_id (TEXT): Identifier for the thread the post belongs to
    - content (TEXT): The text content of the post
    - posted_date_time (TIMESTAMP WITH TIME ZONE): When the post was created
    - channel_name (TEXT): The channel or board where the post was made
    - author (TEXT): Username or identifier of the post author
    - inserted_at (TIMESTAMP WITH TIME ZONE): When the record was added to the database
    """
    
    def __init__(self, agent: KnowledgeAgent):
        """
        Initialize the LLMSQLGenerator.
        
        Args:
            agent: KnowledgeAgent instance for LLM operations
        """
        self.agent = agent
        # Cache for similar queries
        self._query_cache = {}
        # Cache TTL in seconds (5 minutes)
        self._cache_ttl = 300
        # Initialize templates
        self._templates = self._setup_templates()
    
    def _setup_templates(self) -> Dict:
        """Set up the SQL query templates with patterns and parameter generators."""
        templates = {
            "time_based": {
                "last_hour": {
                    "patterns": ["last hour", "past hour", "recent hour"],
                    "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC",
                    "params": [lambda: datetime.now(pytz.UTC) - timedelta(hours=1)]
                },
                "today": {
                    "patterns": ["today", "this day"],
                    "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC",
                    "params": [lambda: datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)]
                },
                "yesterday": {
                    "patterns": ["yesterday", "previous day"],
                    "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s AND posted_date_time < %s ORDER BY posted_date_time DESC",
                    "params": [
                        lambda: (datetime.now(pytz.UTC) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
                        lambda: datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
                    ]
                },
                "last_n_hours": {
                    "pattern": r"last (\d+) hours?",
                    "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC",
                    "param_extractor": lambda match: datetime.now(pytz.UTC) - timedelta(hours=int(match.group(1)))
                },
                "last_n_days": {
                    "pattern": r"last (\d+) days?",
                    "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC",
                    "param_extractor": lambda match: datetime.now(pytz.UTC) - timedelta(days=int(match.group(1)))
                },
                "this_week": {
                    "patterns": ["this week", "current week"],
                    "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC",
                    "params": [lambda: (datetime.now(pytz.UTC) - timedelta(days=datetime.now(pytz.UTC).weekday())).replace(hour=0, minute=0, second=0, microsecond=0)]
                },
                "last_week": {
                    "patterns": ["last week", "past week"],
                    "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC",
                    "params": [lambda: datetime.now(pytz.UTC) - timedelta(days=7)]
                },
                "last_month": {
                    "patterns": ["last month", "past month"],
                    "sql": "SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC",
                    "params": [lambda: datetime.now(pytz.UTC) - timedelta(days=30)]
                }
            },
            "content_based": {
                "pattern": r"(?:about|containing|with|mentioning) ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
                "sql": "SELECT * FROM complete_data WHERE content ILIKE %s ORDER BY posted_date_time DESC",
                "param_extractor": lambda match: f"%{match.group(1).strip()}%"
            },
            "that_contains": {
                "pattern": r"that contains? ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
                "sql": "SELECT * FROM complete_data WHERE content ILIKE %s ORDER BY posted_date_time DESC",
                "param_extractor": lambda match: f"%{match.group(1).strip()}%"
            },
            "author_based": {
                "pattern": r"by (?:author )?([a-zA-Z0-9_\s]+?)(?:\sabout|\scontaining|\swith|\smentioning|\sfrom|\sin|\son|\sduring|\slast|\spast|$)",
                "sql": "SELECT * FROM complete_data WHERE author ILIKE %s ORDER BY posted_date_time DESC",
                "param_extractor": lambda match: f"%{match.group(1).strip()}%"
            },
            "channel_based": {
                "pattern": r"(?:from|in|on) (?:board|channel) ([a-zA-Z0-9_\s]+)",
                "sql": "SELECT * FROM complete_data WHERE channel_name ILIKE %s ORDER BY posted_date_time DESC",
                "param_extractor": lambda match: f"%{match.group(1).strip()}%"
            },
            "random": {
                "pattern": r"random",
                "sql": "SELECT * FROM complete_data ORDER BY RANDOM()",
                "params": []
            },
            "recent": {
                "patterns": ["recent", "latest", "newest"],
                "sql": "SELECT * FROM complete_data ORDER BY posted_date_time DESC",
                "params": []
            },
            "default": {
                "sql": "SELECT * FROM complete_data ORDER BY posted_date_time DESC LIMIT 50",
                "params": []
            }
        }
        return templates
    
    def _get_generation_prompt(self) -> str:
        """
        Get the prompt for SQL generation.
        
        Returns:
            String prompt with schema and instructions
        """
        return f"""You are an expert SQL developer tasked with converting natural language queries into parameterized PostgreSQL queries.

{self.SCHEMA_DEFINITION}

CRITICAL INSTRUCTIONS:
1. Generate ONLY a SELECT query, no other SQL operations are allowed
2. Use %s placeholders for all dynamic values (never interpolate values directly)
3. Always add "ORDER BY posted_date_time DESC" to sort results by recency
4. If a time filter is specified, add it to the WHERE clause
5. IMPORTANT: If the query mentions "containing", "contains", "about", "with", or "mentioning" ANY term or keyword, ALWAYS add a content ILIKE filter:
   - Example: "containing bitcoin" should generate "content ILIKE %s" with parameter "%bitcoin%"
   - The content filter should be combined with other filters using AND 
6. If an author filter is specified, use ILIKE for partial matching
7. If a channel filter is specified, use ILIKE for partial matching
8. For time-based filtering, refer to the current time as NOW() in your explanation
9. Consider pagination and performance by adding LIMIT statements when appropriate
10. PLATFORM-AGNOSTIC HANDLING: When query mentions social media platforms (X, Twitter, Facebook, etc.), treat it as a general content search
    - For example: "trending topics on X" means "content containing popular topics generally"
    - Do NOT require the platform to be explicitly in the data schema
11. For queries about "trending" or "popular" topics, simply return recent content ordered by recency
12. EXPLAIN your reasoning but PUT THE FINAL SQL AS THE LAST LINE in the response

QUERY PARSING GUIDE:
- Time filters: Look for "last hour", "last 3 hours", "today", "this week", etc.
- Content filters: Look for "containing X", "about Y", "with Z", "contains", "mentioning"
- Check specifically for phrases like "that contains X" or "containing X" and ensure X is included as a content filter
- Platform references: When query contains "on X", "on Twitter", "on Facebook", etc., interpret as general content queries, not specific platform filters

RESPONSE FORMAT:
1. First provide your reasoning
2. Then provide the final SQL query as the last line of your response

EXAMPLES:

Query: "Show posts from the last hour"
Reasoning: We need posts from the last hour, so we'll filter where posted_date_time is greater than or equal to 1 hour ago from the current time. We'll use a placeholder for the timestamp value.
SELECT * FROM complete_data WHERE posted_date_time >= %s ORDER BY posted_date_time DESC LIMIT 50

Query: "Find messages containing bitcoin by author satoshi"
Reasoning: We need to search for posts containing "bitcoin" from author "satoshi", using case-insensitive matching for both.
SELECT * FROM complete_data WHERE content ILIKE %s AND author ILIKE %s ORDER BY posted_date_time DESC LIMIT 50

Query: "Find 5 rows from the last 6 hours that contains bitcoin"
Reasoning: We need to find posts from the last 6 hours that contain the word "bitcoin". We'll use a time filter for the 6-hour constraint and a content filter for "bitcoin", limiting to 5 results.
SELECT * FROM complete_data WHERE posted_date_time >= %s AND content ILIKE %s ORDER BY posted_date_time DESC LIMIT 5

Query: "Show posts from channel tech about AI from this month"
Reasoning: We need posts with channel_name containing "tech", content containing "AI", and posted this month (starting from the first day of the current month).
SELECT * FROM complete_data WHERE channel_name ILIKE %s AND content ILIKE %s AND posted_date_time >= %s ORDER BY posted_date_time DESC LIMIT 50

Query: "What are the current trending topics on X?"
Reasoning: This is asking for recent content without specific platform restrictions. Since we're looking for "trending" content, we'll return recent posts ordered by recency, which is our best proxy for trending content in this schema.
SELECT * FROM complete_data ORDER BY posted_date_time DESC LIMIT 50

Query: "Find discussions about crypto on Twitter from yesterday"
Reasoning: This is asking for content containing "crypto" from yesterday. The reference to "Twitter" is treated as general context, not a specific filter on platform.
SELECT * FROM complete_data WHERE content ILIKE %s AND posted_date_time >= %s AND posted_date_time < %s ORDER BY posted_date_time DESC LIMIT 50
"""

    def _get_validation_prompt(self) -> str:
        """
        Get the prompt for SQL validation.
        
        Returns:
            String prompt with validation instructions
        """
        return f"""You are a security expert tasked with validating SQL queries for safety and correctness.

{self.SCHEMA_DEFINITION}

Analyze the following SQL query and check for:
1. SQL injection vulnerabilities
2. Non-SELECT operations (these are not allowed)
3. Missing parameterization (all dynamic values should use %s placeholders)
4. Table access outside of complete_data
5. Syntax errors or logical problems
6. Completeness - ensure the SQL properly implements all filters from the original query:
   - Time filters (if mentioned in the original query)
   - Content filters (if mentioned in the original query - look for terms like "containing", "about", "with", "mentioning")
   - Limit constraints (if mentioned in the original query)

Format your response as a JSON object with the following fields:
- is_safe: Boolean indicating if the query passes all security checks
- reason: String explanation if the query is not safe, otherwise "Safe"
- corrected_sql: String with a corrected version if needed, otherwise the original query
- identified_parameters: Array of parameter descriptions based on placeholders
- missing_filters: Array of filter types missing from the SQL (e.g., ["content_filter"])
"""

    async def generate_sql(
        self, 
        nl_query: str, 
        provider: Optional[ModelProvider] = None,
        use_hybrid_approach: bool = True
    ) -> Tuple[str, List[Any]]:
        """
        Convert a natural language query to parameterized SQL.
        
        Args:
            nl_query: Natural language query string
            provider: Optional model provider parameter (ignored, static providers are always used)
            use_hybrid_approach: Whether to try template matching before LLM (default: True)
            
        Returns:
            Tuple of (sql_query, params) where:
                - sql_query is a string with parameterized SQL
                - params is a list of parameter values
                
        Raises:
            NLQueryParsingError: If the query cannot be parsed or is invalid
        """
        # Check cache first
        cache_key = nl_query.strip().lower()
        if cache_key in self._query_cache:
            cache_entry = self._query_cache[cache_key]
            # Check if cache is still valid
            if (datetime.now(pytz.UTC) - cache_entry["timestamp"]).total_seconds() < self._cache_ttl:
                logger.info(f"Using cached SQL for query: {nl_query[:50]}...")
                return cache_entry["sql"], self._evaluate_parameters(cache_entry["param_generators"])
        
        # Log the query for debugging
        logger.info(f"Generating SQL for query: {nl_query}")
        
        # Check for content terms to ensure they're detected
        content_term = self._extract_content_term(nl_query)
        if content_term:
            logger.info(f"Detected content term in query: '{content_term}'")
        
        # Try template matching first if hybrid approach is enabled
        if use_hybrid_approach:
            try:
                sql, params, param_generators = await self._generate_sql_from_templates(nl_query)
                if sql:
                    logger.info(f"Used template matching for query: {nl_query[:50]}...")
                    logger.info(f"Generated SQL with templates: {sql}")
                    logger.info(f"Parameters: {params}")
                    
                    # Add LIMIT if not present
                    if "LIMIT" not in sql:
                        limit = self._extract_limit(nl_query)
                        if limit:
                            sql += f" LIMIT {limit}"
                            
                    # Cache the result
                    self._query_cache[cache_key] = {
                        "sql": sql,
                        "param_generators": param_generators,
                        "timestamp": datetime.now(pytz.UTC)
                    }
                    return sql, params
            except Exception as e:
                logger.warning(f"Template matching failed: {e}, falling back to LLM")
        
        # Three-stage LLM pipeline with static providers
        # Stage 1: Enhance instructions using PROVIDER_ENHANCER
        logger.info(f"Enhancing query instructions for: {nl_query[:50]}...")
        enhanced_instructions = await self._enhance_instructions(nl_query)
        
        # Stage 2: Generate SQL using PROVIDER_GENERATOR
        logger.info(f"Generating SQL with enhanced instructions...")
        sql_query = await self._generate_raw_sql(enhanced_instructions, self.PROVIDER_GENERATOR)
        
        # Stage 3: Validate the generated SQL using same PROVIDER_GENERATOR
        logger.info(f"Validating generated SQL...")
        validation = await self._validate_sql(sql_query, self.PROVIDER_GENERATOR, nl_query)
        
        if not validation["is_safe"]:
            logger.warning(f"SQL validation failed: {validation['reason']}")
            
            # Check if we have a corrected SQL from validation
            if "corrected_sql" in validation and validation["corrected_sql"]:
                logger.info(f"Using corrected SQL from validation: {validation['corrected_sql']}")
                sql_query = validation["corrected_sql"]
            else:
                # Try with template-based fallback
                try:
                    default_sql, default_params, default_generators = await self._generate_sql_from_templates(nl_query, fallback_mode=True)
                    
                    # Cache the result
                    self._query_cache[cache_key] = {
                        "sql": default_sql,
                        "param_generators": default_generators,
                        "timestamp": datetime.now(pytz.UTC)
                    }
                    
                    return default_sql, default_params
                except Exception as e:
                    logger.error(f"Template fallback also failed: {e}")
                    raise NLQueryParsingError(f"Generated SQL failed safety checks: {validation['reason']}")
        
        # Use corrected SQL if provided
        sql_query = validation.get("corrected_sql", sql_query)
        
        # Add LIMIT if not present
        if "LIMIT" not in sql_query.upper():
            limit = self._extract_limit(nl_query)
            if limit:
                sql_query += f" LIMIT {limit}"
            else:
                sql_query += " LIMIT 50"  # Default limit
        
        # Extract parameters
        params, param_generators = await self._extract_parameters(nl_query, sql_query, validation.get("identified_parameters", []))
        
        # Cache the result
        self._query_cache[cache_key] = {
            "sql": sql_query,
            "param_generators": param_generators,
            "timestamp": datetime.now(pytz.UTC)
        }
        
        return sql_query, params

    async def _generate_combined_template_sql(self, nl_query: str) -> Optional[Tuple[str, List[Any], List[Callable]]]:
        """
        Generate SQL query combining time and content filters from templates.
        
        This handles the case where a query has both time and content filters.
        
        Args:
            nl_query: Natural language query
            
        Returns:
            Optional tuple of (sql_query, params, param_generators)
        """
        nl_lower = nl_query.lower()
        
        # Extract time filter
        time_params = []
        time_generators = []
        time_condition = None
        
        # Check for time filter
        # Last hour
        if any(p in nl_lower for p in ["last hour", "past hour", "recent hour"]):
            time_condition = "posted_date_time >= %s"
            hour_ago = datetime.now(pytz.UTC) - timedelta(hours=1)
            time_params.append(hour_ago)
            time_generators.append(lambda: datetime.now(pytz.UTC) - timedelta(hours=1))
        
        # Last X hours
        hour_match = re.search(r"last (\d+) hours?", nl_lower)
        if hour_match:
            hours = int(hour_match.group(1))
            time_condition = "posted_date_time >= %s"
            hours_ago = datetime.now(pytz.UTC) - timedelta(hours=hours)
            time_params.append(hours_ago)
            hours_copy = hours  # Create a copy for the lambda
            time_generators.append(lambda h=hours_copy: datetime.now(pytz.UTC) - timedelta(hours=h))
            
        # If no time filter found, return None
        if not time_condition:
            return None
            
        # Extract content filter
        content_params = []
        content_generators = []
        content_condition = None
        
        # Extract content term
        content_term = self._extract_content_term(nl_query)
        if content_term:
            content_condition = "content ILIKE %s"
            content_params.append(f"%{content_term}%")
            term_copy = content_term  # Create a copy for lambda
            content_generators.append(lambda t=term_copy: f"%{t}%")
        
        # If no content filter found, return None
        if not content_condition:
            return None
            
        # Combine conditions
        sql = f"SELECT * FROM complete_data WHERE {time_condition} AND {content_condition} ORDER BY posted_date_time DESC"
        
        # Extract limit
        limit = self._extract_limit(nl_lower)
        if limit:
            sql += f" LIMIT {limit}"
        
        # Combine parameters and generators
        params = time_params + content_params
        generators = time_generators + content_generators
        
        logger.info(f"Generated combined template SQL: {sql}")
        logger.info(f"Combined parameters: {params}")
        
        return sql, params, generators
        
    async def _generate_sql_from_templates(self, nl_query: str, fallback_mode: bool = False) -> Tuple[str, List[Any], List[Callable]]:
        """
        Unified approach for generating SQL from predefined templates.
        
        This method combines the previous _try_template_matching and simple_replit_sql_generator
        approaches into a more comprehensive solution.
        
        Args:
            nl_query: Natural language query
            fallback_mode: If True, use more lenient matching and always return a result
            
        Returns:
            Tuple of (sql_query, params, param_generators)
            
        Raises:
            ValueError: If no template matches and not in fallback mode
        """
        nl_lower = nl_query.lower()
        limit = self._extract_limit(nl_lower)
        
        # Try to generate combined template SQL for queries with both time and content filters
        combined_result = await self._generate_combined_template_sql(nl_query)
        if combined_result:
            logger.info("Using combined template SQL for time + content filters")
            return combined_result
        
        # Check for random ordering first, as it's a higher priority pattern
        is_random = "random" in nl_lower
        
        # Get time-related parameters if present
        time_params = []
        time_generators = []
        has_time_filter = False
        
        # Check for specific time windows in the query
        if "last week" in nl_lower or "past week" in nl_lower:
            week_ago = datetime.now(pytz.UTC) - timedelta(days=7)
            time_params.append(week_ago)
            time_generators.append(lambda: datetime.now(pytz.UTC) - timedelta(days=7))
            has_time_filter = True
        elif "last month" in nl_lower or "past month" in nl_lower:
            month_ago = datetime.now(pytz.UTC) - timedelta(days=30)
            time_params.append(month_ago)
            time_generators.append(lambda: datetime.now(pytz.UTC) - timedelta(days=30))
            has_time_filter = True
        elif "last hour" in nl_lower or "past hour" in nl_lower or "recent hour" in nl_lower:
            hour_ago = datetime.now(pytz.UTC) - timedelta(hours=1)
            time_params.append(hour_ago)
            time_generators.append(lambda: datetime.now(pytz.UTC) - timedelta(hours=1))
            has_time_filter = True
        else:
            # Check for regex-based time matches
            hour_match = re.search(r"last (\d+) hours?", nl_lower)
            if hour_match:
                hours = int(hour_match.group(1))
                hours_ago = datetime.now(pytz.UTC) - timedelta(hours=hours)
                time_params.append(hours_ago)
                hours_copy = hours  # Create a copy for lambda
                time_generators.append(lambda h=hours_copy: datetime.now(pytz.UTC) - timedelta(hours=h))
                has_time_filter = True
            else:
                day_match = re.search(r"last (\d+) days?", nl_lower)
                if day_match:
                    days = int(day_match.group(1))
                    days_ago = datetime.now(pytz.UTC) - timedelta(days=days)
                    time_params.append(days_ago)
                    days_copy = days  # Create a copy for lambda
                    time_generators.append(lambda d=days_copy: datetime.now(pytz.UTC) - timedelta(days=d))
                    has_time_filter = True
                else:
                    has_time_filter = False
        
        # Check for content filter
        content_params = []
        content_generators = []
        has_content_filter = False
        
        # First check for "that contains" pattern since it's causing issues
        that_contains_template = self._templates["that_contains"]
        that_contains_match = re.search(that_contains_template["pattern"], nl_lower)
        if that_contains_match:
            content_param = that_contains_template["param_extractor"](that_contains_match)
            content_params.append(content_param)
            param_copy = content_param  # Create a copy for capture
            content_generators.append(lambda p=param_copy: p)
            has_content_filter = True
            logger.info(f"Matched 'that contains' template with parameter: {content_param}")
        else:
            # Check standard content template
            content_template = self._templates["content_based"]
            content_match = re.search(content_template["pattern"], nl_lower)
            if content_match:
                content_param = content_template["param_extractor"](content_match)
                content_params.append(content_param)
                param_copy = content_param  # Create a copy for capture
                content_generators.append(lambda p=param_copy: p)
                has_content_filter = True
                logger.info(f"Matched content template with parameter: {content_param}")
        
        # Handle random ordering with time and/or content filters
        if is_random:
            sql_parts = ["SELECT * FROM complete_data"]
            params = []
            generators = []
            
            # Add WHERE clause if we have filters
            where_clauses = []
            
            if has_time_filter:
                where_clauses.append("posted_date_time >= %s")
                params.extend(time_params)
                generators.extend(time_generators)
            
            if has_content_filter:
                where_clauses.append("content ILIKE %s")
                params.extend(content_params)
                generators.extend(content_generators)
            
            if where_clauses:
                sql_parts.append("WHERE " + " AND ".join(where_clauses))
            
            # Add random ordering
            sql_parts.append("ORDER BY RANDOM()")
            
            # Combine SQL parts
            sql = " ".join(sql_parts)
            
            # Add limit if specified, otherwise use default of 10 for random
            if limit:
                sql = self._add_limit_to_sql(sql, limit)
            else:
                sql += " LIMIT 10"  # Default limit for random
                
            return sql, params, generators
        
        # If not random, proceed with normal template matching
        
        # Try time-based templates first
        time_templates = self._templates["time_based"]
        for template_name, template in time_templates.items():
            # Check for pattern-based matches
            if "patterns" in template and any(pattern in nl_lower for pattern in template["patterns"]):
                sql = template["sql"]
                params = [param_func() for param_func in template["params"]]
                generators = template["params"]
                
                if limit:
                    sql = self._add_limit_to_sql(sql, limit)
                    
                return sql, params, generators
            
            # Check for regex-based matches
            if "pattern" in template:
                match = re.search(template["pattern"], nl_lower)
                if match:
                    sql = template["sql"]
                    
                    # Extract parameters using the provided function
                    if "param_extractor" in template:
                        param = template["param_extractor"](match)
                        params = [param]
                        # Create a generator that produces the same parameter
                        param_copy = param  # Create a copy for capture
                        generators = [lambda p=param_copy: p]
                    else:
                        params = [param_func() for param_func in template.get("params", [])]
                        generators = template.get("params", [])
                    
                    if limit:
                        sql = self._add_limit_to_sql(sql, limit)
                        
                    return sql, params, generators
        
        # Try content-based template
        content_template = self._templates["content_based"]
        content_match = re.search(content_template["pattern"], nl_lower)
        if content_match:
            sql = content_template["sql"]
            param = content_template["param_extractor"](content_match)
            params = [param]
            # Create a generator that produces the same parameter
            param_copy = param  # Create a copy for capture
            generators = [lambda p=param_copy: p]
            
            if limit:
                sql = self._add_limit_to_sql(sql, limit)
                
            # Check for author in the same query
            author_template = self._templates["author_based"]
            author_match = re.search(author_template["pattern"], nl_lower)
            if author_match:
                # Add author condition
                sql = sql.replace("ORDER BY", f"AND author ILIKE %s ORDER BY")
                author_param = author_template["param_extractor"](author_match)
                params.append(author_param)
                author_copy = author_param  # Create a copy
                generators.append(lambda a=author_copy: a)
            
            return sql, params, generators
        
        # Try author-based template
        author_template = self._templates["author_based"]
        author_match = re.search(author_template["pattern"], nl_lower)
        if author_match:
            sql = author_template["sql"]
            param = author_template["param_extractor"](author_match)
            params = [param]
            # Create a generator
            param_copy = param
            generators = [lambda p=param_copy: p]
            
            if limit:
                sql = self._add_limit_to_sql(sql, limit)
                
            return sql, params, generators
        
        # Try channel-based template
        channel_template = self._templates["channel_based"]
        channel_match = re.search(channel_template["pattern"], nl_lower)
        if channel_match:
            sql = channel_template["sql"]
            param = channel_template["param_extractor"](channel_match)
            params = [param]
            # Create a generator
            param_copy = param
            generators = [lambda p=param_copy: p]
            
            if limit:
                sql = self._add_limit_to_sql(sql, limit)
                
            return sql, params, generators
        
        # Try recent template
        recent_template = self._templates["recent"]
        if any(pattern in nl_lower for pattern in recent_template["patterns"]):
            sql = recent_template["sql"]
            params = recent_template["params"]
            generators = params
            
            if limit:
                sql = self._add_limit_to_sql(sql, limit)
            else:
                sql += " LIMIT 20"  # Default limit for recent
                
            return sql, params, generators
        
        # If in fallback mode, return the default template
        if fallback_mode:
            default = self._templates["default"]
            sql = default["sql"]
            if limit:
                sql = sql.replace("LIMIT 50", f"LIMIT {limit}")
            return sql, default["params"], default["params"]
        
        # No template match and not in fallback mode
        return None, None, None

    def _add_limit_to_sql(self, sql: str, limit: int) -> str:
        """Add or replace LIMIT clause in SQL query."""
        if "LIMIT" in sql:
            # Replace existing limit
            return re.sub(r"LIMIT \d+", f"LIMIT {limit}", sql)
        else:
            # Add new limit
            return f"{sql} LIMIT {limit}"

    def _extract_limit(self, nl_query: str, default: Optional[int] = None) -> Optional[int]:
        """Extract limit parameter from query text."""
        text = nl_query.lower()
        patterns = [
            r"\blimit\s+(\d+)\b",
            r"\btop\s+(\d+)\b",
            r"\bfirst\s+(\d+)\b",
            r"\b(\d+)\s+(?:random\s+)?(?:recent\s+)?(?:latest\s+)?(?:newest\s+)?(?:rows|results|posts|threads|messages)\b",
        ]
        for pattern in patterns:
            limit_match = re.search(pattern, text)
            if limit_match:
                return int(limit_match.group(1))
        return default

    async def _enhance_instructions(self, nl_query: str) -> str:
        """
        Enhance natural language query with structured instructions.
        """
        # Try to get prompt and character_slug from YAML
        sql_prompts = self.agent.prompts.get('sql_prompts', {})
        enhance_cfg = sql_prompts.get('enhance_instructions', {})
        prompt_template = enhance_cfg.get('system_prompt')
        character_slug = enhance_cfg.get('character_slug', self.CHARACTER_ENHANCER)
        if not prompt_template:
            prompt_template = """You are an expert in understanding database query intent.\nAnalyze this natural language query and enhance it with specific details:\n\n1. Identify all time-based filters (last hour, today, etc.)\n2. Identify content filters (contains X, about Y)\n3. Identify author or channel filters\n4. Structure these identified elements clearly\n5. Preserve ALL original query intent"""
        prompt = f"{prompt_template}\n\nOriginal query: {nl_query}\n\nEnhanced instructions:"
        try:
            enhanced = await self.agent.generate_chunks(
                content=prompt,
                provider=self.PROVIDER_ENHANCER,
                character_slug=get_venice_character_slug(character_slug=character_slug)
            )
            if isinstance(enhanced, dict) and "chunks" in enhanced and enhanced["chunks"]:
                enhanced_text = enhanced["chunks"][0]
            else:
                enhanced_text = str(enhanced)
            return f"{enhanced_text.strip()}\n\nOriginal query: {nl_query}"
        except Exception as e:
            logger.warning(f"Error enhancing query: {e}, using original query")
            return nl_query

    async def _generate_raw_sql(
        self, 
        instructions: str, 
        provider: ModelProvider
    ) -> str:
        """
        Generate raw SQL from instructions using LLM.
        """
        sql_prompts = self.agent.prompts.get('sql_prompts', {})
        gen_cfg = sql_prompts.get('generate_sql', {})
        prompt_template = gen_cfg.get('system_prompt')
        character_slug = gen_cfg.get('character_slug', self.CHARACTER_GENERATOR)
        if not prompt_template:
            prompt_template = self._get_generation_prompt()
        prompt = f"{prompt_template}\n\nQuery: {instructions}\n\nReasoning:"
        try:
            response = await self.agent.generate_chunks(
                content=prompt,
                provider=provider,
                character_slug=get_venice_character_slug(character_slug=character_slug)
            )
            if isinstance(response, dict) and "chunks" in response and response["chunks"]:
                response_text = response["chunks"][0]
            else:
                response_text = str(response)
            lines = response_text.strip().split('\n')
            sql_line = lines[-1].strip()
            if not sql_line.upper().startswith("SELECT"):
                for line in reversed(lines):
                    if line.upper().strip().startswith("SELECT"):
                        sql_line = line.strip()
                        break
                else:
                    raise NLQueryParsingError("No valid SELECT statement found in LLM response")
            return sql_line
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise NLQueryParsingError(f"Failed to generate SQL: {e}")

    async def _validate_sql(
        self, 
        sql_query: str,
        provider: ModelProvider,
        nl_query: str = None
    ) -> Dict[str, Any]:
        """
        Validate SQL for security and correctness using LLM.
        """
        if not sql_query.upper().strip().startswith("SELECT"):
            return {
                "is_safe": False,
                "reason": "Only SELECT statements are allowed",
                "corrected_sql": None
            }
        for pattern in ["--", ";", "DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER"]:
            if pattern in sql_query.upper():
                return {
                    "is_safe": False,
                    "reason": f"Potential SQL injection detected: '{pattern}'",
                    "corrected_sql": None
                }
        if nl_query:
            nl_lower = nl_query.lower()
            content_terms = ["contain", "about", "with", "mention"]
            has_content_reference = any(term in nl_lower for term in content_terms)
            if has_content_reference and "content ILIKE" not in sql_query.lower():
                logger.warning(f"Content filter mentioned in query but missing in SQL: {nl_query}")
        sql_prompts = self.agent.prompts.get('sql_prompts', {})
        val_cfg = sql_prompts.get('validate_sql', {})
        prompt_template = val_cfg.get('system_prompt')
        character_slug = val_cfg.get('character_slug', self.CHARACTER_VALIDATOR)
        if not prompt_template:
            prompt_template = self._get_validation_prompt()
        if nl_query:
            prompt = f"{prompt_template}\n\nOriginal Query: {nl_query}\nSQL Query: {sql_query}"
        else:
            prompt = f"{prompt_template}\n\nSQL Query: {sql_query}"
        try:
            response = await self.agent.generate_chunks(
                content=prompt,
                provider=provider,
                character_slug=get_venice_character_slug(character_slug=character_slug)
            )
            if isinstance(response, dict) and "chunks" in response and response["chunks"]:
                response_text = response["chunks"][0]
            else:
                response_text = str(response)
            json_match = re.search(r'{[\s\S]*}', response_text)
            if not json_match:
                logger.warning(f"Could not find JSON in validation response: {response_text}")
                return self._simple_validation_fallback(sql_query, nl_query)
            validation_json = json_match.group(0)
            validation = json.loads(validation_json)
            if "is_safe" not in validation:
                logger.warning(f"Validation response missing 'is_safe' field: {validation}")
                validation["is_safe"] = False
                validation["reason"] = "Invalid validation response"
            if "missing_filters" in validation and validation["missing_filters"]:
                logger.warning(f"Missing filters detected: {validation['missing_filters']}")
                if "content_filter" in validation["missing_filters"] and nl_query:
                    content_term = self._extract_content_term(nl_query)
                    if content_term and "corrected_sql" in validation:
                        corrected_sql = validation["corrected_sql"]
                        if "WHERE" in corrected_sql:
                            corrected_sql = corrected_sql.replace("WHERE", f"WHERE content ILIKE '%{content_term}%' AND")
                        else:
                            corrected_sql = corrected_sql.replace("ORDER BY", f"WHERE content ILIKE '%{content_term}%' ORDER BY")
                        validation["corrected_sql"] = corrected_sql
                        logger.info(f"Corrected SQL with missing content filter: {corrected_sql}")
            return validation
        except Exception as e:
            logger.error(f"Error validating SQL: {e}")
            return self._simple_validation_fallback(sql_query, nl_query)

    def _extract_content_term(self, nl_query: str) -> Optional[str]:
        """Extract content search term from natural language query."""
        nl_lower = nl_query.lower()
        # Improved regex patterns with non-greedy matching and lookahead to avoid capturing time/other filters
        content_patterns = [
            r"containing ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"about ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"with ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"mentions ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"mentioning ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"contains ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"that contains ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"that contain ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"has ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"having ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"including ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)",
            r"related to ([a-zA-Z0-9_\s]+?)(?:\sfrom|\sin|\son|\sduring|\slast|\spast|\sby|$)"
        ]
        
        # First try pattern matching with improved non-greedy patterns
        for pattern in content_patterns:
            content_match = re.search(pattern, nl_lower)
            if content_match:
                extracted_term = content_match.group(1).strip()
                # Remove common stop words and trailing words that might have been included
                stop_words = ["the", "a", "an", "in", "on", "at", "from", "to", "by"]
                for word in stop_words:
                    if extracted_term.endswith(f" {word}"):
                        extracted_term = extracted_term[:-(len(word) + 1)].strip()
                
                logger.info(f"Extracted content term using pattern '{pattern}': '{extracted_term}'")
                return extracted_term
        
        # If no direct pattern match, try looking for keywords after content indicators
        content_indicators = ["about", "contain", "with", "mention", "has", "have", "include", "related"]
        for indicator in content_indicators:
            if indicator in nl_lower:
                # Find the position of the indicator
                pos = nl_lower.find(indicator) + len(indicator)
                # Get the text after the indicator
                text_after = nl_lower[pos:].strip()
                if text_after:
                    # Extract the main term before any time/other filters
                    main_term = re.split(r'\sfrom|\sin|\son|\sduring|\slast|\spast|\sby', text_after)[0].strip()
                    if main_term:
                        # Split into words and filter out stop words
                        words = main_term.split()
                        # Skip common connectors
                        skip_words = ["to", "the", "a", "an", "with", "for", "of", "in", "on", "by"]
                        term_words = []
                        for word in words[:3]:  # Consider first 3 words at most
                            if word not in skip_words:
                                term_words.append(word)
                            if len(term_words) == 2:  # Limit to 2 meaningful words
                                break
                        
                        if term_words:
                            extracted_term = " ".join(term_words)
                            logger.info(f"Extracted content term using indicator '{indicator}': '{extracted_term}'")
                            return extracted_term
        
        # Special case for "that contains/contain" pattern
        if "that contain" in nl_lower:
            # Get everything after "that contain" but before time filters
            parts = re.split(r"that contain\s+", nl_lower, 1)
            if len(parts) > 1:
                text_after = parts[1]
                # Split at time filters or other common delimiters
                main_term = re.split(r'\sfrom|\sin|\son|\sduring|\slast|\spast|\sby', text_after)[0].strip()
                if main_term:
                    # Extract first meaningful word
                    words = main_term.split()
                    skip_words = ["s", "the", "a", "an", "with", "for", "of", "in", "on", "by"]
                    for word in words:
                        if word not in skip_words:
                            logger.info(f"Extracted content term from 'that contain': '{word}'")
                            return word
        
        return None

    def _simple_validation_fallback(self, sql_query: str, nl_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Simple validation fallback when LLM validation fails.
        
        Args:
            sql_query: SQL query to validate
            nl_query: Original natural language query
            
        Returns:
            Dictionary with validation results
        """
        try:
            if sqlparse is None:
                sql_clean = sql_query.strip().rstrip(";")
                sql_upper = sql_clean.upper()
                if not sql_upper.startswith("SELECT"):
                    return {"is_safe": False, "reason": "Only SELECT statements are allowed"}
                if "FROM COMPLETE_DATA" not in sql_upper and "FROM PUBLIC.COMPLETE_DATA" not in sql_upper:
                    return {"is_safe": False, "reason": "Query must use the complete_data table"}
                if "%s" not in sql_query and not ("SELECT * FROM complete_data" in sql_query and "LIMIT" in sql_query):
                    return {"is_safe": False, "reason": "No parameter placeholders found"}
                param_count = sql_query.count("%s")
                identified_parameters = [f"Parameter {i+1}" for i in range(param_count)]
                result = {
                    "is_safe": True,
                    "reason": "Safe",
                    "corrected_sql": sql_query,
                    "identified_parameters": identified_parameters
                }
                if nl_query:
                    content_term = self._extract_content_term(nl_query)
                    if content_term and "content ILIKE" not in sql_query.lower():
                        result["missing_filters"] = ["content_filter"]
                return result

            # Parse the SQL query
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return {"is_safe": False, "reason": "Failed to parse SQL"}
            
            stmt = parsed[0]
            
            # Check statement type
            if stmt.get_type().upper() != "SELECT":
                return {"is_safe": False, "reason": "Only SELECT statements are allowed"}
            
            # Check for placeholders
            if "%s" not in sql_query and not ("SELECT * FROM complete_data" in sql_query and "LIMIT" in sql_query and "%s" not in sql_query):
                return {"is_safe": False, "reason": "No parameter placeholders found"}
            
            # Check that it's using the correct table
            tables = [t.get_name() for t in stmt.get_sublists() if hasattr(t, 'get_name')]
            if "complete_data" not in tables and "public.complete_data" not in tables:
                return {"is_safe": False, "reason": "Query must use the complete_data table"}
            
            # Count parameters for identified_parameters
            param_count = sql_query.count("%s")
            identified_parameters = [f"Parameter {i+1}" for i in range(param_count)]
            
            # Check for missing content filter if a content search term is mentioned
            missing_filters = []
            if nl_query:
                content_term = self._extract_content_term(nl_query)
                if content_term and "content ILIKE" not in sql_query.lower():
                    missing_filters.append("content_filter")
            
            result = {
                "is_safe": True,
                "reason": "Safe",
                "corrected_sql": sql_query,
                "identified_parameters": identified_parameters
            }
            
            if missing_filters:
                result["missing_filters"] = missing_filters
                
                # Try to correct the SQL if content filter is missing
                if "content_filter" in missing_filters:
                    content_term = self._extract_content_term(nl_query)
                    if content_term:
                        # Add content filter to SQL
                        if "WHERE" in sql_query:
                            # Add content filter to existing WHERE clause
                            corrected_sql = sql_query.replace("WHERE", f"WHERE content ILIKE '%{content_term}%' AND")
                        else:
                            # Add new WHERE clause with content filter before ORDER BY
                            corrected_sql = sql_query.replace("ORDER BY", f"WHERE content ILIKE '%{content_term}%' ORDER BY")
                        
                        result["corrected_sql"] = corrected_sql
            
            return result
            
        except Exception as e:
            logger.error(f"Error in simple SQL validation: {e}")
            # Assume it's unsafe if we can't validate
            return {"is_safe": False, "reason": f"Validation error: {e}"}

    async def _extract_parameters(
        self, 
        nl_query: str, 
        sql_query: str, 
        param_descriptions: List[str]
    ) -> Tuple[List[Any], List[Any]]:
        """
        Extract parameters from natural language and SQL query.
        
        Args:
            nl_query: Natural language query
            sql_query: SQL query with placeholders
            param_descriptions: List of parameter descriptions
            
        Returns:
            Tuple of (parameter_values, parameter_generators)
        """
        nl_lower = nl_query.lower()
        sql_lower = sql_query.lower()
        params = []
        param_generators = []
        now = datetime.now(pytz.UTC)
        
        # Count how many parameters we need
        param_count = sql_query.count("%s")
        
        # If no parameters needed, return empty lists
        if param_count == 0:
            return [], []
        
        # Extract content filter parameters first if present
        content_filter_added = False
        if "content ilike %s" in sql_lower:
            content_term = self._extract_content_term(nl_query)
            if content_term:
                params.append(f"%{content_term}%")
                term_copy = content_term  # Create a copy for lambda
                param_generators.append(lambda t=term_copy: f"%{t}%")
                content_filter_added = True
        
        # Extract time-based parameters
        if "posted_date_time >= %s" in sql_query and len(params) < param_count:
            # Last hour
            if "last hour" in nl_lower or "past hour" in nl_lower or "recent hour" in nl_lower:
                hour_ago = now - timedelta(hours=1)
                params.append(hour_ago)
                param_generators.append(lambda: datetime.now(pytz.UTC) - timedelta(hours=1))
                
            # Last X hours
            hour_match = re.search(r"last (\d+) hours?", nl_lower)
            if hour_match:
                hours = int(hour_match.group(1))
                if hours <= 0:
                    hours = 1  # Default to 1 hour if invalid
                hours_ago = now - timedelta(hours=hours)
                params.append(hours_ago)
                hours_copy = hours  # Create a copy for the lambda
                param_generators.append(lambda h=hours_copy: datetime.now(pytz.UTC) - timedelta(hours=h))
                
            # Today
            elif "today" in nl_lower:
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                params.append(today_start)
                param_generators.append(lambda: datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0))
                
            # This week
            elif "this week" in nl_lower or "current week" in nl_lower:
                week_start = now - timedelta(days=now.weekday())
                week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                params.append(week_start)
                param_generators.append(lambda: (datetime.now(pytz.UTC) - timedelta(days=datetime.now(pytz.UTC).weekday())).replace(hour=0, minute=0, second=0, microsecond=0))
                
            # Last week
            elif "last week" in nl_lower or "past week" in nl_lower:
                week_ago = now - timedelta(days=7)
                params.append(week_ago)
                param_generators.append(lambda: datetime.now(pytz.UTC) - timedelta(days=7))
                
            # Last month
            elif "last month" in nl_lower or "past month" in nl_lower:
                month_ago = now - timedelta(days=30)
                params.append(month_ago)
                param_generators.append(lambda: datetime.now(pytz.UTC) - timedelta(days=30))
                
            # Last X days
            day_match = re.search(r"last (\d+) days?", nl_lower)
            if day_match:
                days = int(day_match.group(1))
                if days <= 0:
                    days = 1  # Default to 1 day if invalid
                days_ago = now - timedelta(days=days)
                params.append(days_ago)
                days_copy = days  # Create a copy for the lambda
                param_generators.append(lambda d=days_copy: datetime.now(pytz.UTC) - timedelta(days=d))
        
        # Yesterday needs both start and end times
        if "yesterday" in nl_lower and "posted_date_time >= %s" in sql_query and "posted_date_time < %s" in sql_query:
            yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Add parameters in the correct order
            params = []
            param_generators = []
            
            params.append(yesterday_start)
            param_generators.append(lambda: (datetime.now(pytz.UTC) - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))
            
            params.append(yesterday_end)
            param_generators.append(lambda: datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0))
        
        # Content search (if not already added)
        if "content ilike %s" in sql_lower and not content_filter_added and len(params) < param_count:
            content_term = self._extract_content_term(nl_query)
            if content_term:
                params.append(f"%{content_term}%")
                term_copy = content_term  # Create a copy for the lambda
                param_generators.append(lambda t=term_copy: f"%{t}%")
            else:
                # Fallback for when we can't extract the content term directly
                content_words = []
                # Extract potential keywords (words longer than 3 chars that aren't common words)
                common_words = ["find", "show", "get", "give", "select", "from", "with", "about", 
                                "containing", "contains", "that", "rows", "posts", "messages", 
                                "results", "last", "hour", "hours", "today", "week", "month"]
                
                for word in nl_lower.split():
                    clean_word = re.sub(r'[^a-zA-Z0-9]', '', word)
                    if len(clean_word) > 3 and clean_word.lower() not in common_words:
                        content_words.append(clean_word)
                
                if content_words:
                    # Use the longest word as it's more likely to be a significant term
                    best_word = max(content_words, key=len)
                    params.append(f"%{best_word}%")
                    word_copy = best_word  # Create a copy for the lambda
                    param_generators.append(lambda w=word_copy: f"%{w}%")
                else:
                    # If all else fails, use a very permissive filter
                    params.append("%%")
                    param_generators.append(lambda: "%%")
        
        # Author search
        if "author ilike %s" in sql_lower and len(params) < param_count:
            author_match = re.search(r"by (?:author )?([a-zA-Z0-9_\s]+?)(?:\sabout|\scontaining|\swith|\smentioning|\sfrom|\sin|\son|\sduring|\slast|\spast|$)", nl_lower)
            if author_match:
                author = author_match.group(1).strip()
                params.append(f"%{author}%")
                author_copy = author  # Create a copy for the lambda
                param_generators.append(lambda a=author_copy: f"%{a}%")
        
        # Channel search
        if "channel_name ilike %s" in sql_lower and len(params) < param_count:
            channel_match = re.search(r"(?:from|in|on) (?:board|channel) ([a-zA-Z0-9_\s]+)", nl_lower)
            if channel_match:
                channel = channel_match.group(1).strip()
                params.append(f"%{channel}%")
                channel_copy = channel  # Create a copy for the lambda
                param_generators.append(lambda c=channel_copy: f"%{c}%")
        
        # If we still don't have enough parameters, add default parameters based on descriptions
        while len(params) < param_count:
            # If we have a description, use it to try to infer the parameter
            if len(param_descriptions) > len(params):
                desc = param_descriptions[len(params)].lower()
                if "date" in desc or "time" in desc:
                    # Default to 24 hours ago for time parameters
                    params.append(now - timedelta(hours=24))
                    param_generators.append(lambda: datetime.now(pytz.UTC) - timedelta(hours=24))
                elif "content" in desc or "text" in desc:
                    # Extract potential content search terms
                    content_term = self._extract_content_term(nl_query)
                    if content_term:
                        params.append(f"%{content_term}%")
                        term_copy = content_term  # Create a copy for the lambda
                        param_generators.append(lambda t=term_copy: f"%{t}%")
                    else:
                        # If no content term found, use a default
                        params.append("%%")
                        param_generators.append(lambda: "%%")
                elif "limit" in desc:
                    # Extract limit parameter
                    limit = self._extract_limit(nl_lower, default=50)
                    params.append(limit)
                    limit_copy = limit  # Create a copy for the lambda
                    param_generators.append(lambda l=limit_copy: l)
                else:
                    # Generic parameter
                    params.append(None)
                    param_generators.append(lambda: None)
            else:
                # No description available
                params.append(None)
                param_generators.append(lambda: None)
        
        # Log the parameters for debugging
        logger.debug(f"Extracted parameters for SQL query:")
        for i, param in enumerate(params):
            logger.debug(f"  Param {i+1}: {param}")
        
        # Ensure we have exactly the right number of parameters
        return params[:param_count], param_generators[:param_count]

    def _evaluate_parameters(self, param_generators: List[Any]) -> List[Any]:
        """
        Evaluate parameter generator functions to get current values.
        
        Args:
            param_generators: List of parameter generator functions
            
        Returns:
            List of parameter values
        """
        return [generator() for generator in param_generators]

    def get_query_description(self, nl_query: str) -> Dict[str, Union[str, datetime]]:
        """
        Generate a human-readable description of the natural language query.
        
        Args:
            nl_query: Natural language query string
            
        Returns:
            Dictionary with descriptive information about the query
        """
        nl_lower = nl_query.lower()
        now = datetime.now(pytz.UTC)
        description = {
            "original_query": nl_query,
            "query_time": now.isoformat(),
            "filters": []
        }
        
        # Track if we found an explicit time filter
        time_filter_found = False
        
        # Detect time-based filters
        if any(p in nl_lower for p in ["last hour", "past hour", "previous hour", "recent hour"]):
            description["time_filter"] = "Last hour"
            description["filters"].append("Time: Last hour")
            time_filter_found = True
            
        hour_match = re.search(r"last (\d+) hours?", nl_lower)
        if hour_match and not time_filter_found:
            hours = int(hour_match.group(1))
            if hours <= 0:
                description["time_filter"] = "Invalid time range"
                description["filters"].append("Time: Invalid range (must be positive)")
            else:
                description["time_filter"] = f"Last {hours} hours"
                description["filters"].append(f"Time: Last {hours} hours")
            time_filter_found = True
        
        if any(p in nl_lower for p in ["today", "this day"]) and not time_filter_found:
            description["time_filter"] = "Today"
            description["filters"].append("Time: Today")
            time_filter_found = True
            
        if any(p in nl_lower for p in ["yesterday", "previous day"]) and not time_filter_found:
            description["time_filter"] = "Yesterday"
            description["filters"].append("Time: Yesterday")
            time_filter_found = True
            
        day_match = re.search(r"last (\d+) days?", nl_lower)
        if day_match and not time_filter_found:
            days = int(day_match.group(1))
            if days <= 0:
                description["time_filter"] = "Invalid time range"
                description["filters"].append("Time: Invalid range (must be positive)")
            else:
                description["time_filter"] = f"Last {days} days"
                description["filters"].append(f"Time: Last {days} days")
            time_filter_found = True
        
        # Detect limit
        limit = self._extract_limit(nl_lower)
        if limit:
            description["limit"] = limit
            description["filters"].append(f"Limit: {limit} rows")
        
        # Detect random
        if "random" in nl_lower:
            description["random"] = True
            description["filters"].append("Random ordering")
        
        # Detect content filters using improved extraction
        content_term = self._extract_content_term(nl_query)
        if content_term:
            description["content_filter"] = content_term
            description["filters"].append(f"Content: Contains '{content_term}'")
        
        # Detect author filter
        author_match = re.search(r"by (?:author )?([a-zA-Z0-9_\s]+?)(?:\sabout|\scontaining|\swith|\smentioning|\sfrom|\sin|\son|\sduring|\slast|\spast|$)", nl_lower)
        if author_match:
            author = author_match.group(1).strip()
            if author:
                description["author_filter"] = author
                description["filters"].append(f"Author: '{author}'")
        
        # Detect channel filter
        channel_patterns = [
            r"from (?:board|channel) (.+?)(?:\s|$)",
            r"in (?:board|channel) (.+?)(?:\s|$)",
            r"on (?:board|channel) (.+?)(?:\s|$)"
        ]
        
        for pattern in channel_patterns:
            channel_match = re.search(pattern, nl_lower)
            if channel_match:
                channel_name = channel_match.group(1).strip()
                if channel_name:
                    description["channel_filter"] = channel_name
                    description["filters"].append(f"Channel: '{channel_name}'")
                    break
        
        # If no time filters were detected, add the default
        if not time_filter_found:
            description["time_filter"] = "Last 24 hours (default)"
            description["filters"].append("Time: Last 24 hours (default)")
        
        return description
