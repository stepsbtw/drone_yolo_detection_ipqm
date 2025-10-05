# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies for headless OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create output directory
RUN mkdir -p output

# Set the default command
ENTRYPOINT ["python", "src/main.py"]