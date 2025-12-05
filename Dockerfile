FROM python:3.13-slim

# Install system dependencies (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only dependency files first (better caching)
COPY pyproject.toml .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy the actual package code
COPY . .

# Install the package into the container
RUN pip install --no-cache-dir .

# Default command
CMD ["bash"]
