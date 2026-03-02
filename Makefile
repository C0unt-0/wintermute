# Makefile — Wintermute hybrid development orchestration
#
# Hybrid architecture: Docker runs infrastructure (PostgreSQL + Redis),
# native processes run on Apple Silicon with full MLX access.
#
# Usage:
#   make dev       Start full dev stack (infra + API + worker + web)
#   make infra     Start Docker infrastructure only
#   make api       Start FastAPI server natively (requires infra)
#   make worker    Start Celery worker natively (requires infra)
#   make web       Start Vite dev server (requires api)
#   make stop      Stop all processes
#   make clean     Stop everything and remove Docker volumes
#   make prod      Start full Docker stack (no MLX, for CI/Linux)

.PHONY: dev infra api worker web stop clean prod check-venv check-port migrate help

# ── Configuration ─────────────────────────────────────────────────────
SHELL := /bin/bash
VENV := ./venv/bin

# Load .env if present
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Defaults (overridden by .env or CLI: make dev API_PORT=8001)
WINTERMUTE_DATABASE_URL ?= postgresql+psycopg://wintermute:wintermute_dev@localhost:5432/wintermute
REDIS_URL ?= redis://localhost:6379/0
API_PORT ?= 8000

# ── Help (default target) ────────────────────────────────────────────
help: ## Show available targets
	@grep -Eh '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Checks ───────────────────────────────────────────────────────────
check-venv:
	@test -f $(VENV)/python || { echo "ERROR: venv not found at $(VENV)."; \
		echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -e '.[all]'"; \
		exit 1; }

check-port: ## Verify API_PORT is available
	@lsof -i :$(API_PORT) -sTCP:LISTEN >/dev/null 2>&1 && { \
		echo ""; \
		echo "ERROR: Port $(API_PORT) is already in use:"; \
		lsof -i :$(API_PORT) -sTCP:LISTEN | head -5; \
		echo ""; \
		echo "Fix: make dev API_PORT=8001  (or any free port)"; \
		echo ""; \
		exit 1; \
	} || true

# ── Infrastructure ───────────────────────────────────────────────────
infra: ## Start Docker infrastructure (PostgreSQL + Redis)
	@if nc -z localhost 6379 2>/dev/null && nc -z localhost 5432 2>/dev/null; then \
		echo "Infrastructure already running (Redis :6379, PostgreSQL :5432)."; \
	else \
		docker compose up -d && bash scripts/wait-for-pg.sh localhost 5432 30; \
	fi

# ── Migrations ───────────────────────────────────────────────────────
migrate: check-venv infra ## Run Alembic migrations against PostgreSQL
	WINTERMUTE_DATABASE_URL=$(WINTERMUTE_DATABASE_URL) \
		$(VENV)/alembic upgrade head

# ── Native services ─────────────────────────────────────────────────
api: check-venv check-port ## Start FastAPI server natively (with MLX)
	WINTERMUTE_DATABASE_URL=$(WINTERMUTE_DATABASE_URL) \
	REDIS_URL=$(REDIS_URL) \
		$(VENV)/uvicorn api.main:app --reload --host 0.0.0.0 --port $(API_PORT)

worker: check-venv ## Start Celery worker natively (with MLX)
	REDIS_URL=$(REDIS_URL) \
	WINTERMUTE_DATABASE_URL=$(WINTERMUTE_DATABASE_URL) \
		$(VENV)/celery -A src.wintermute.engine.worker worker -l info

web: ## Start Vite dev server with HMR
	cd web && VITE_API_PORT=$(API_PORT) npm run dev

# ── Full dev stack ───────────────────────────────────────────────────
dev: check-venv infra migrate check-port ## Start full dev stack (infra + API + worker + web)
	@echo ""
	@echo "=========================================="
	@echo "  Wintermute Dev Stack"
	@echo "=========================================="
	@echo "  PostgreSQL :5432 | Redis :6379 (Docker)"
	@echo "  API        :$(API_PORT) (native, MLX enabled)"
	@echo "  Worker     (native, MLX enabled)"
	@echo "  Web UI     :5173 (Vite + HMR)"
	@echo "=========================================="
	@echo "  Press Ctrl+C to stop all services"
	@echo "=========================================="
	@echo ""
	@WINTERMUTE_DATABASE_URL=$(WINTERMUTE_DATABASE_URL) \
	 REDIS_URL=$(REDIS_URL) \
		$(VENV)/uvicorn api.main:app --reload --host 0.0.0.0 --port $(API_PORT) & \
	 REDIS_URL=$(REDIS_URL) \
	 WINTERMUTE_DATABASE_URL=$(WINTERMUTE_DATABASE_URL) \
		$(VENV)/celery -A src.wintermute.engine.worker worker -l info & \
	 trap 'kill %1 %2 2>/dev/null; echo ""; echo "Dev stack stopped (infra containers still running)."' EXIT INT TERM; \
	 cd web && VITE_API_PORT=$(API_PORT) npm run dev

# ── Production (full Docker) ────────────────────────────────────────
prod: ## Start full Docker stack (for CI / non-Mac platforms)
	docker compose --profile full up --build

# ── Cleanup ──────────────────────────────────────────────────────────
stop: ## Stop all running services
	-docker compose stop
	-pkill -f "uvicorn api.main:app" 2>/dev/null || true
	-pkill -f "celery.*wintermute" 2>/dev/null || true

clean: stop ## Stop everything and remove Docker volumes
	docker compose down -v
