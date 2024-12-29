# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire application first
COPY . .

# Install poetry and dependencies
ENV PATH="/root/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Create base directories
RUN mkdir -p data logs

# Expose port
EXPOSE 5000

# Run the application with optimized Gunicorn settings
CMD ["poetry", "run", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "600", \
     "--workers", "1", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--worker-tmp-dir", "/dev/shm", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--log-level", "debug", \
     "app:app"]