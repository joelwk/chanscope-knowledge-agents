# Natural Language Query Examples

This document provides examples of how to use the natural language query endpoint with curl commands.

## Basic Time-Based Queries

### Last Hour

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Give me threads from the last hour",
    "limit": 10
  }'
```

### Last N Hours

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts from the last 5 hours",
    "limit": 20
  }'
```

### Today

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get threads from today",
    "limit": 15
  }'
```

### Yesterday

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts from yesterday",
    "limit": 15
  }'
```

### Last Week

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get threads from last week",
    "limit": 25
  }'
```

## Content Search Queries

### Keyword Search

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find messages containing bitcoin",
    "limit": 10
  }'
```

### Combined with Time Filter

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts from yesterday containing crypto",
    "limit": 15
  }'
```

## Author Queries

### Filter by Author

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find posts by author john",
    "limit": 10
  }'
```

### Author + Time + Content

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get posts by author alice from last week containing AI",
    "limit": 15
  }'
```

## Channel/Board Queries

### Filter by Channel

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show threads from board tech",
    "limit": 10
  }'
```

### Channel + Content + Time

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find posts from channel biz about crypto from this month",
    "limit": 20
  }'
```

## Complex Combined Queries

### All Filters

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show posts by author satoshi from board crypto containing bitcoin from last week",
    "limit": 15
  }'
```

### Default Filter when None Specified

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find messages about machine learning",
    "limit": 10,
    "use_default_time_filter": true
  }'
```

## Disable Default Time Filter

If you want to retrieve all matching records without a time filter:

```bash
curl -X POST "http://localhost/api/v1/nl_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show all posts containing python",
    "limit": 50,
    "use_default_time_filter": false
  }'
```

Note: When `use_default_time_filter` is set to `false`, the system will not apply the default 24-hour time filter if no explicit time filter is found in the query. This is useful when you want to search across all time periods without restriction. 