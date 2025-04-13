# Stratification Guide: Best Practices & Refresh Patterns

## Overview

The stratification process is a critical component of the Chanscope system that creates representative samples from the complete dataset. This guide explains how stratification works, when samples are refreshed, and best practices for maintaining up-to-date stratified samples.

## Understanding Stratification

### What is Stratification?

Stratification is the process of creating a smaller, representative subset of the complete dataset that preserves the distribution of important features. In Chanscope, stratification:

1. Reduces computational requirements for embedding and query processing
2. Maintains representative samples across different time periods and boards
3. Enables more efficient semantic search and analysis
4. Balances data representation for more accurate query results

### Stratification Metadata

Each stratified sample includes metadata with:
- Column information
- Datetime column specifications
- Row count (typically 1000 rows)
- Timestamp of when the stratification was performed

Example metadata:
```json
{
  "columns": ["id", "thread_id", "content", "posted_date_time", "channel_name", "author", "inserted_at"],
  "datetime_columns": ["posted_date_time", "inserted_at"],
  "row_count": 1000,
  "timestamp": "2025-04-09T14:28:47.255119"
}
```

## Stratification Refresh Behavior

### When Stratified Samples Are Updated

By default, the system preserves existing stratified samples to save computational resources. A stratified sample is only regenerated when:

1. No stratified sample exists
2. The `--force-refresh` flag is explicitly provided
3. An incremental refresh with the `--force-refresh` flag is scheduled

The system will **not** automatically regenerate the stratified sample just because new data has been added. This is an intentional design choice to optimize for performance, especially in resource-constrained environments.

### Log Indicators

You can identify stratification refresh behavior in the logs:

- **Using existing sample**: `Using existing stratified sample with 1000 rows (force_refresh=False)`
- **Regenerating sample**: `Will regenerate stratified sample and embeddings`

## Best Practices for Stratification Management

### 1. Regular Forced Refreshes

For production environments, schedule regular forced refreshes to ensure your stratified samples reflect current data:

```bash
# Refresh data and regenerate stratified sample daily
poetry run python scripts/scheduled_update.py refresh --force-refresh --continuous --interval=86400
```

### 2. Two-Stage Refresh for Resource Constraints

In memory-constrained environments, use a two-stage approach:

```bash
# First: Regenerate stratified sample only
poetry run python scripts/scheduled_update.py refresh --force-refresh --skip-embeddings

# Later (during off-peak hours): Generate embeddings
poetry run python scripts/scheduled_update.py embeddings
```

### 3. Monitoring Stratification Age

Check the age of your stratified sample:

```bash
poetry run python scripts/scheduled_update.py status
```

Look for the timestamp in the stratified metadata file to determine when it was last regenerated. A warning sign is when the timestamp is significantly older than your most recent data.

### 4. Custom Refresh Intervals

For advanced use cases, consider implementing different refresh intervals:

```bash
# Process new data hourly, regenerate stratified sample daily
# Note: This requires a custom implementation using scheduled tasks
0 * * * * poetry run python scripts/scheduled_update.py refresh
0 0 * * * poetry run python scripts/scheduled_update.py refresh --force-refresh
```

## Troubleshooting

### Common Issues

1. **Outdated stratified sample**: If query results don't reflect recent data despite data processing, your stratified sample may be outdated. Use `--force-refresh` to regenerate it.

2. **Missing data in query results**: Check if the data was properly ingested AND if the stratified sample was regenerated after ingestion.

3. **Timestamp discrepancies**: If the stratified_metadata.json timestamp is old compared to your recent data, regenerate the sample using `--force-refresh`.

### Verification Steps

To verify your stratification is current:

1. Check the timestamp in the stratified_metadata.json file
2. Run a system status check: `scheduled_update.py status`
3. Examine logs for the message "Using existing stratified sample" vs. "Regenerating stratified sample"

## Conclusion

The stratification process is designed to balance computational efficiency with data currency. By understanding when and how stratified samples are refreshed, you can ensure your Chanscope deployment maintains accurate and relevant data representations for query processing.

Remember: **The `--force-refresh` flag is your primary tool for ensuring stratified samples stay current with newly ingested data.** 