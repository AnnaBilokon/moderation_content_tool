FROM python:3.11-slim

# Optional build tools for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for better cache)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the rest of the backend source
COPY . .

EXPOSE 8000
# For Render, no host/port env needed — just listen on 0.0.0.0:8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
