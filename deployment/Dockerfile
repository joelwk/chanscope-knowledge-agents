# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1 \
    && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev

# Runtime stage
FROM python:3.11-slim as app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories with proper permissions
RUN mkdir -p /app/data/stratified /app/logs /app/temp_files \
    && chown -R nobody:nogroup /app \
    && chmod -R 755 /app \
    && chmod 777 /app/data \
    && chmod -R 777 /app/data/stratified \
    && chmod 777 /app/logs \
    && chmod 777 /app/temp_files

# Copy setup script first and make it executable
COPY --chown=nobody:nogroup deployment/setup.sh /app/deployment/
RUN chmod +x /app/deployment/setup.sh

# Copy application code
COPY --chown=nobody:nogroup . .

# Set minimal environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DOCKER_ENV=true \
    TZ=UTC

EXPOSE 80

USER nobody

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost/api/v1/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "/app/deployment/setup.sh"]