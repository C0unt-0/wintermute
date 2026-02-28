# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Python + static files
FROM python:3.11-slim
WORKDIR /app
COPY --from=frontend /app/web/dist ./web/dist
COPY pyproject.toml ./
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
RUN pip install --no-cache-dir -e ".[api]"
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
