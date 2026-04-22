# Use Python 3.11 for stability with TensorFlow/XGBoost
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for TA-Lib and other ML libs
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
