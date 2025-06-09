FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for tika, tesseract, and other libraries
RUN apt-get update && apt-get install -y \
    default-jre-headless \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Run as a non-root user
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose the port for the web server
EXPOSE ${PORT}

# Start command
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT}
