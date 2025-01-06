# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies and SSL certificates
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using pip (more reliable than installer script)
RUN pip install --no-cache-dir poetry==1.7.1 && \
    poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Development stage
FROM builder AS development

# Install all dependencies including development
RUN poetry install --no-root --with dev

# Copy application code
COPY . .

# Create base directories
RUN mkdir -p data logs temp_files

# Install the application
RUN poetry install --only-root

# Set only non-dynamic environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Create startup script for handling different service types
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting service: $SERVICE_TYPE in development mode"\n\
\n\
if [ "$SERVICE_TYPE" = "api" ]; then\n\
    echo "Starting API server..."\n\
    exec poetry run python -m flask --app api.app run --host 0.0.0.0 --port 5000 --debug\n\
elif [ "$SERVICE_TYPE" = "ui" ]; then\n\
    echo "Starting Chainlit UI..."\n\
    cd /app && exec poetry run python -m chainlit run chainlit_frontend/app.py --host 0.0.0.0 --port 8000\n\
else\n\
    echo "ERROR: Please set SERVICE_TYPE to either api or ui"\n\
    exit 1\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Expose ports for both Flask API and Chainlit UI
EXPOSE 5000 8000

# Set the startup script as the entrypoint
ENTRYPOINT ["/bin/bash", "/app/start.sh"]

# Production stage
FROM builder AS production

# Install only production dependencies
RUN poetry install --no-root --without dev

# Copy application code
COPY . .

# Create base directories
RUN mkdir -p data logs temp_files

# Install the application
RUN poetry install --only-root

# Set only non-dynamic environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Create startup script for handling different service types
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting service: $SERVICE_TYPE in production mode"\n\
\n\
if [ "$SERVICE_TYPE" = "api" ]; then\n\
    echo "Starting API server with gunicorn..."\n\
    exec poetry run gunicorn \
        --bind 0.0.0.0:5000 \
        --timeout 1200 \
        --graceful-timeout 300 \
        --keep-alive 300 \
        --workers 1 \
        --threads 4 \
        --worker-class gthread \
        --worker-tmp-dir /dev/shm \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --log-level ${LOG_LEVEL:-info} \
        api.app:app\n\
elif [ "$SERVICE_TYPE" = "ui" ]; then\n\
    echo "Starting Chainlit UI..."\n\
    cd /app && exec poetry run python -m chainlit run chainlit_frontend/app.py --host 0.0.0.0 --port 8000\n\
else\n\
    echo "ERROR: Please set SERVICE_TYPE to either api or ui"\n\
    exit 1\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Expose ports for both Flask API and Chainlit UI
EXPOSE 5000 8000

# Set the startup script as the entrypoint
ENTRYPOINT ["/bin/bash", "/app/start.sh"]