#!/usr/bin/env bash
# scripts/wait-for-pg.sh — Wait for PostgreSQL to accept connections
set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-5432}"
MAX_RETRIES="${3:-30}"

echo "Waiting for PostgreSQL at ${HOST}:${PORT}..."
for i in $(seq 1 "$MAX_RETRIES"); do
    if command -v pg_isready >/dev/null 2>&1; then
        if pg_isready -h "$HOST" -p "$PORT" -U wintermute >/dev/null 2>&1; then
            echo "PostgreSQL is ready."
            exit 0
        fi
    elif nc -z "$HOST" "$PORT" 2>/dev/null; then
        echo "PostgreSQL port is open."
        exit 0
    fi
    echo "  Attempt $i/$MAX_RETRIES — not ready yet..."
    sleep 1
done

echo "ERROR: PostgreSQL did not become ready in time."
exit 1
