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

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root

# Runtime stage
FROM python:3.11-slim as app

# Copy Python packages from builder
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

# Create home directory and data directories with proper permissions for nobody:nogroup user
RUN mkdir -p /home/nobody && \
    mkdir -p /app/data /app/data/stratified /app/data/shared /app/logs /app/temp_files \
    && chown -R nobody:nogroup /app /home/nobody \
    && chmod -R 777 /app/data \
    && chmod -R 777 /app/data/stratified \
    && chmod -R 777 /app/data/shared \
    && chmod -R 777 /app/logs \
    && chmod -R 777 /app/temp_files \
    && chmod -R 777 /home/nobody

# Copy setup script first and make it executable
COPY --chown=nobody:nogroup deployment/setup.sh /app/deployment/
RUN chmod +x /app/deployment/setup.sh

# Copy application code
COPY --chown=nobody:nogroup . .

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DOCKER_ENV=true \
    REPLIT_ENV=false \
    HOME=/home/nobody \
    TZ=UTC

EXPOSE 80

# Run as non-root user
USER nobody

# Health check with extended start period to account for data loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost/api/v1/health || exit 1

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "/app/deployment/setup.sh"]