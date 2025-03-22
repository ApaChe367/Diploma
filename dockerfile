# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (if required)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
RUN pip install --no-cache-dir pandas pvlib matplotlib

# Run the script when the container launches
CMD ["python", "Solar_Model_Output.py"]
