# Python runtime for the Qualia Bet Market web app.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Render (and most PaaS) inject the port to listen on via $PORT.
# Fall back to 5000 for local `docker run`.
ENV PORT=5000
EXPOSE 5000

# Shell form so $PORT expands at runtime.
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000}
