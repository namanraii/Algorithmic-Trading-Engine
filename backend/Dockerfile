# Use Python 3.11 for stability with TensorFlow/XGBoost
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TA-Lib and other ML libs
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code into a 'backend' folder to preserve package structure
COPY . ./backend

# Expose port
EXPOSE 8000

# Start the application using the package path
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
