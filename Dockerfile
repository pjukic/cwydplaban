# Base image
FROM python:3.11-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    CHROMADB_PATH=/app/vectorstores/db/chroma/ \
    FAISSDB_PATH=/app/vectorstores/db/faiss/ \
    DATA_PATH=/app/data/

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    wget \
    curl \
    unzip \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Expose port for Chainlit
EXPOSE 8012

# Set entrypoint for the application
ENTRYPOINT ["chainlit", "run"]
CMD ["app.py", "--host", "0.0.0.0", "--port", "8012"]
