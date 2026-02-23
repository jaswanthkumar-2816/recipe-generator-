"""WSGI entrypoint for Render/Gunicorn.

Render typically runs `gunicorn app:app` by default. This module exposes
`app` so the service can start correctly.
"""

from Foodimg2Ing import app  # noqa: F401
