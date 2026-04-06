# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and PyTorch (if needed)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights during build phase
RUN python download_models.py

# Make port 10000 available to the world outside this container
EXPOSE 10000

# Set environment variables
ENV FLASK_APP=run.py
ENV PORT=10000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "200", "run:app"]
