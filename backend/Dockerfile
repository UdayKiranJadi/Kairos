FROM python:3.11-slim

WORKDIR /app

# System deps needed for psycopg, torch, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy the rest of the app
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
