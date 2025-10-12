FROM python:3.12-slim

WORKDIR /backend

# Copy requirements (create this file first)
COPY docker_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r docker_requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["python", "transformer_app.py"]