# Use Node 20 LTS to avoid Node 22 issues with Next.js dev on Windows
FROM node:20-alpine

WORKDIR /app

# Install deps using lockfile for reproducibility
COPY frontend/package*.json ./
RUN npm ci

# Copy source (for dev we also mount it, so this is just a baseline)
COPY frontend/ .

EXPOSE 3000

# Start Next.js in dev mode (hot reload)
CMD ["npm", "run", "dev"]
