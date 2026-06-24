import os

# Gunicorn configuration file to override Render defaults
# This ensures that even if Render runs its default start command,
# Gunicorn will use the correct timeout and worker settings.

# Timeout for workers (in seconds)
# Increased to 300s to allow for slow CPU model loading and inference
timeout = 300

# Number of workers
# Restricted to 1 to prevent multiple workers from running out of memory (512MB RAM limit)
workers = 1

# Bind address and port
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Enable debug logs
loglevel = "debug"
