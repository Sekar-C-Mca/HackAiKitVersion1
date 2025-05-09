# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and pyproject.toml
COPY pyproject.toml ./

# Install dependencies
RUN pip install --upgrade pip build setuptools wheel
RUN pip install . --no-cache-dir

# Copy the application code
COPY . .

# Create a non-root user for security
RUN useradd -m hackaiuser
RUN chown -R hackaiuser:hackaiuser /app
USER hackaiuser

# Expose the port
EXPOSE $PORT

# Command to run the application
CMD uvicorn hackaikit.api.main:app --host 0.0.0.0 --port $PORT 