# LLM-Based Natural Language to SQL Feature

## Overview

The LLM-Based Natural Language to SQL feature enables users to query the database using everyday language rather than SQL. This implementation uses a sophisticated three-stage LLM architecture to convert natural language queries like "Show me threads from the last hour" into safe, parameterized SQL queries.

## Architecture

The implementation follows a three-stage architecture:

### Stage 1: Instruction Enhancement (PROVIDER_ENHANCER)

- **Input:** Natural language query (e.g., "Find posts about Bitcoin from last week")
- **Process:** First LLM analyzes the query, identifies key elements (time filters, content filters, etc.), and reformulates it into detailed instructions
- **Output:** Enhanced instructions with structured details, preserving the original query

### Stage 2: SQL Generation (PROVIDER_GENERATOR)

- **Input:** Enhanced instructions from Stage 1
- **Process:** Second LLM converts the structured instructions into appropriate PostgreSQL
- **Output:** Raw SQL query with parameter placeholders

### Stage 3: SQL Validation (Same PROVIDER_GENERATOR)

- **Input:** Generated SQL query
- **Process:** Same LLM as Stage 2 validates SQL for security, correctness, and best practices
- **Output:** Validated or corrected SQL with parameter descriptions

### Stage 4: Parameter Extraction

- **Input:** Original natural language query + validated SQL
- **Process:** Extract parameters from natural language and SQL structure using rule-based logic
- **Output:** SQL query with bound parameters

### Hybrid Approach for Efficiency

The system uses a hybrid approach for optimal performance:

1. **Template Matching:** Common query patterns are matched against templates for near-instant responses
2. **Three-Stage LLM Pipeline:** Complex or novel queries are handled by the full LLM pipeline
3. **Result Caching:** Similar queries are cached to improve response times

## Implementation Details

### Core Components

1. **LLMSQLGenerator Class**
   - Handles the entire process of converting NL to SQL
   - Maintains query cache for improved performance
   - Implements template matching, instruction enhancement, SQL generation and validation
   - Uses static provider configurations for different stages:
     - `PROVIDER_ENHANCER` (default: OpenAI) for instruction enhancement
     - `PROVIDER_GENERATOR` (default: Venice) for SQL generation and validation

2. **LLM Prompts**
   - Carefully designed prompts for each stage:
     - Instruction enhancement: Extracts structured details from natural language
     - SQL generation: Includes schema definitions and clear guidelines
     - SQL validation: Focuses on security and correctness validation
   - Examples covering common query patterns

3. **Venice Character Slug Support**
   - Uses the `get_venice_character_slug()` utility function
   - Priority order for character slug:
     1. Explicitly provided character slug
     2. Environment variable (`VENICE_CHARACTER_SLUG`)
     3. Configuration class (`Config.get_venice_character_slug()`)
     4. Default value (`pisagor-ai`)

## Integration Details

### Database Schema

```sql
CREATE TABLE complete_data (
    id SERIAL PRIMARY KEY,
    thread_id TEXT,
    content TEXT,
    posted_date_time TIMESTAMP WITH TIME ZONE,
    channel_name TEXT,
    author TEXT,
    inserted_at TIMESTAMP WITH TIME ZONE
)
```

### API Integration

The feature is exposed via the `/nl_query` endpoint, which accepts the following parameters:

- `query` (required): Natural language query string
- `limit` (optional): Maximum number of results to return
- `provider` (optional): Override the default LLM provider

### Unit Tests

Comprehensive test suite in `tests/llm_sql_test.py`:

1. **Template Matching Tests**: Verify common queries are correctly matched to templates
2. **Instruction Enhancement Tests**: Ensure query details are correctly extracted
3. **SQL Generation Tests**: Check that enhanced instructions produce correct SQL
4. **SQL Validation Tests**: Verify security checks catch potentially unsafe SQL
5. **Parameter Extraction Tests**: Ensure parameters are correctly extracted and bound

## Coding Standards

- **Security**: All SQL is validated for injection vulnerabilities, and only SELECT statements are allowed
- **Efficiency**: Template matching and caching for common queries
- **Robustness**: Fallback validation when LLM validation fails
- **Maintainability**: Clear component separation, consistent logging, and thorough documentation

## Example Usage

```python
from knowledge_agents.model_ops import KnowledgeAgent
from knowledge_agents.llm_sql_generator import LLMSQLGenerator

async def main():
    # Initialize the agent
    agent = await KnowledgeAgent.create()
    
    # Create SQL generator
    sql_generator = LLMSQLGenerator(agent)
    
    # Convert natural language to SQL
    sql_query, params = await sql_generator.generate_sql(
        nl_query="Show posts about Bitcoin from last week",
        use_hybrid_approach=True  # Enable template matching for efficiency
    )
    
    print(f"SQL Query: {sql_query}")
    print(f"Parameters: {params}")
```

## Testing

The implementation includes comprehensive test coverage:

1. **Unit Tests**
   - Tests for template matching
   - Tests for SQL generation
   - Tests for parameter extraction
   - Tests for validation and security

2. **Integration Tests**
   - End-to-end tests for the API
   - Performance benchmarks
   - Error handling tests

## Performance Considerations

1. **Caching**
   - Similar queries are cached with a TTL of 5 minutes
   - Cache includes parameter generators for time-based freshness

2. **Hybrid Approach**
   - Templates for common patterns avoid unnecessary LLM calls
   - Progressive enhancement based on query complexity

3. **Parameter Generators**
   - Time-based parameters use generators to ensure freshness
   - Parameter regeneration without requiring new LLM calls

## Future Enhancements

Potential improvements for future versions:

1. **More Query Capabilities**
   - Support for aggregations (COUNT, AVG, etc.)
   - Support for JOINs with related tables
   - Advanced sorting and grouping options

2. **Performance Optimizations**
   - Query plan analysis for complex queries
   - Distributed caching for high-volume scenarios
   - Precomputed results for common queries

3. **User Experience**
   - Query suggestions based on data patterns
   - Interactive query refinement
   - Visual query builder integration

## Dependencies

- **Knowledge Agent Framework:** Uses the existing KnowledgeAgent for LLM operations
- **PostgreSQL:** Connects to the same Replit PostgreSQL database as other features
- **sqlparse:** Used for fallback SQL validation when LLM validation fails 