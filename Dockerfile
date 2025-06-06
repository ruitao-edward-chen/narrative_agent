# Multi-stage build for efficient image
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install --production=false
COPY frontend/ ./
RUN npm run build

# Python backend
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from previous stage
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Create necessary directories
RUN mkdir -p agents

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "app.py"] 