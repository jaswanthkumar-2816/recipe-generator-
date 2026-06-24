# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies for OpenCV and PyTorch (if needed)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000 as per Hugging Face Spaces guidelines
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy the current directory contents into the container and set ownership to user 1000
COPY --chown=1000:1000 . $HOME/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Make port 10000 available
EXPOSE 10000

# Set environment variables
ENV FLASK_APP=run.py
ENV PORT=10000

# Run model download at container startup (has internet access) and launch gunicorn
CMD python download_models.py && gunicorn --bind 0.0.0.0:10000 --workers 1 --timeout 200 run:app

