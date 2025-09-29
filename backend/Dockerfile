FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Correct: file exists at context root
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Correct: copy the rest of backend source
COPY . /app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

