# Web UI Design — Replace TUI with React Analyst Workstation

**Date:** 2026-02-28
**Status:** Approved
**Supersedes:** 2026-02-27-tui-control-panel-design.md

---

## Goal

Remove the Textual-based TUI entirely and replace it with a React + Vite web UI that serves as an analyst workstation. The web UI mirrors the TUI's six-screen functionality (Dashboard, Scan, Training, Adversarial, Pipeline, Vault) with a browser-native experience. The existing FastAPI backend is extended with new endpoints and a WebSocket for live events.

## Decisions

| Aspect | Decision |
|--------|----------|
| Approach | Monorepo — `web/` directory alongside `api/` and `src/` |
| TUI | Remove entirely, relocate hooks to `engine/` |
| Backend | Extend existing FastAPI with routers + WebSocket |
| Frontend | React 18 + Vite + Tailwind CSS + Recharts |
| Aesthetic | Terminal Noir — dark theme, phosphor green/cyan/red palette |
| Real-time | WebSocket at `/api/v1/ws` for training/adversarial/pipeline events |
| Deploy | Multi-stage Docker build, FastAPI serves built static files |

---

## 1. TUI Removal

### Deleted

- `/src/wintermute/tui/` — entire directory (18 files, ~2,171 lines)
- `/tests/test_tui.py` — TUI integration tests
- `textual>=1.0.0` from `pyproject.toml` optional dependencies
- `wintermute tui` subcommand from `cli.py`

### Relocated

| From | To | Reason |
|------|----|--------|
| Event types (`tui/events.py`) | `engine/events.py` | Transport-agnostic event definitions |
| Hook classes (`tui/hooks.py`) | `engine/hooks.py` | Base hook interface reusable by web + CLI |

### Unchanged

- CLI commands (`wintermute scan`, `train`, `evaluate`, etc.)
- All models, trainers, data pipeline
- Existing FastAPI scan/status endpoints

---

## 2. Backend — Extended FastAPI

### File Structure

```
api/
├── main.py              # Add routers, WebSocket, static file mount
├── schemas.py           # All request/response models
├── routers/
│   ├── scan.py          # Existing scan endpoints extracted here
│   ├── training.py      # POST start, GET status, POST cancel
│   ├── adversarial.py   # POST start, GET status, POST cancel
│   ├── pipeline.py      # POST build/synthetic/pretrain, GET status, POST cancel
│   ├── vault.py         # GET samples, GET sample/{id}
│   └── dashboard.py     # GET system status + metrics
├── ws.py                # WebSocket manager — broadcast live events
└── hooks.py             # Web hook implementations (emit to WebSocket)
```

### Endpoints

| Group | Method | Path | Purpose |
|-------|--------|------|---------|
| Health | GET | `/health` | Existing health check |
| Dashboard | GET | `/api/v1/dashboard` | Model version, F1, accuracy, vault size, family counts |
| Scan | POST | `/api/v1/scan` | Existing — enqueue binary for analysis |
| Scan | GET | `/api/v1/status/{job_id}` | Existing — poll scan job status |
| Training | POST | `/api/v1/training/start` | Start training with config |
| Training | GET | `/api/v1/training/{job_id}/status` | Poll training progress |
| Training | POST | `/api/v1/training/{job_id}/cancel` | Cancel training job |
| Adversarial | POST | `/api/v1/adversarial/start` | Start adversarial loop |
| Adversarial | GET | `/api/v1/adversarial/{job_id}/status` | Poll adversarial progress |
| Adversarial | POST | `/api/v1/adversarial/{job_id}/cancel` | Cancel adversarial job |
| Pipeline | POST | `/api/v1/pipeline/{operation}` | Start build/synthetic/pretrain |
| Pipeline | GET | `/api/v1/pipeline/{job_id}/status` | Poll pipeline progress |
| Pipeline | POST | `/api/v1/pipeline/{job_id}/cancel` | Cancel pipeline job |
| Vault | GET | `/api/v1/vault/samples` | List adversarial samples |
| Vault | GET | `/api/v1/vault/samples/{id}` | Sample detail + diff |

### WebSocket

```
WS /api/v1/ws
```

Broadcasts JSON messages with the same event model as the TUI:

```json
{"type": "epoch_complete", "epoch": 1, "phase": "A", "loss": 0.42, "train_acc": 0.85, "val_acc": 0.83, "f1": 0.81}
{"type": "adversarial_cycle_end", "cycle": 3, "metrics": {"evasion_rate": 0.12, "adv_tpr": 0.94}}
{"type": "pipeline_progress", "operation": "synthetic", "progress": 0.65, "message": "Generated 325/500"}
{"type": "vault_sample_added", "sample": {"id": "abc", "family": "Ramnit", "confidence": 0.23}}
{"type": "activity_log", "text": "Training complete", "level": "ok"}
{"type": "scan_progress", "phase": "disassembly", "data": {}}
```

### Hook Architecture

```python
# engine/hooks.py — base interface
class TrainingHook:
    def on_epoch(self, epoch, phase, loss, train_acc, val_acc, f1, elapsed): ...
    def on_log(self, text, level="info"): ...
    cancelled: bool

# api/hooks.py — web implementation
class WebTrainingHook(TrainingHook):
    def on_epoch(self, ...):
        ws_manager.broadcast({"type": "epoch_complete", ...})
```

---

## 3. Frontend — React + Vite

### File Structure

```
web/
├── index.html
├── package.json
├── vite.config.ts            # Proxy /api/* → localhost:8000 in dev
├── tsconfig.json
├── tailwind.config.ts
├── src/
│   ├── main.tsx
│   ├── App.tsx               # Layout shell + tab routing
│   ├── api/
│   │   ├── client.ts         # Fetch wrapper for REST endpoints
│   │   └── ws.ts             # WebSocket client with reconnect
│   ├── hooks/
│   │   ├── useWebSocket.ts   # React hook for WS events
│   │   ├── useJob.ts         # Start → poll → return state
│   │   └── useDashboard.ts   # Dashboard metrics
│   ├── pages/
│   │   ├── Dashboard.tsx     # Metric cards, family chart, activity log
│   │   ├── Scan.tsx          # File upload, disassembly view, verdict
│   │   ├── Training.tsx      # Config form, epoch table, charts
│   │   ├── Adversarial.tsx   # Red/blue teams, episode log, cycles
│   │   ├── Pipeline.tsx      # Operation selector, progress, log
│   │   └── Vault.tsx         # Sample table, detail + diff
│   ├── components/
│   │   ├── StatCard.tsx
│   │   ├── ConfidenceBar.tsx
│   │   ├── ConfigPanel.tsx   # Right-side config form
│   │   ├── ActivityLog.tsx
│   │   ├── DiffView.tsx
│   │   └── SparklineChart.tsx
│   └── styles/
│       └── theme.css         # CSS custom properties
```

### Page-to-TUI Mapping

| TUI Screen | Web Page | Key Differences |
|------------|----------|-----------------|
| Dashboard | `Dashboard.tsx` | Same stat cards + family chart, activity log via WebSocket |
| Scan | `Scan.tsx` | Drag-and-drop file upload, syntax-highlighted disassembly |
| Training | `Training.tsx` | Side panel config form, epoch table + Recharts sparklines |
| Adversarial | `Adversarial.tsx` | Red/blue cards, streaming episode log via WebSocket |
| Pipeline | `Pipeline.tsx` | Operation selector dropdown, progress bar via WebSocket |
| Vault | `Vault.tsx` | Click-to-expand rows, diff view panel |

### Dependencies

- `react` + `react-dom` (18.x)
- `react-router-dom` (tab routing)
- `tailwindcss` (utility-first styling)
- `recharts` (charts — sparklines, bar charts, family distribution)

---

## 4. Visual Identity — Terminal Noir

### Color Palette

```css
:root {
  --bg-primary: #0a0e14;
  --bg-surface: #111820;
  --bg-elevated: #1a2332;
  --border: #1e2d3d;
  --text-primary: #e0e6ed;
  --text-muted: #6b7d8e;
  --safe: #00e88f;       /* verdicts, success states */
  --threat: #ff3b5c;     /* malicious, errors */
  --data: #00d4ff;       /* data values, links, accent */
  --warn: #ffb224;       /* warnings, amber alerts */
  --purple: #b48ead;     /* call instructions, secondary */
}
```

### Typography

- **Code/data:** JetBrains Mono — opcode displays, tables, metrics
- **Headings:** Space Mono — section titles, tab labels
- **Body:** Outfit — descriptions, log entries, form labels

### Signature Elements

- Scan results animate like a terminal printout — opcodes scroll in with subtle glow
- Confidence bars fill with a liquid gradient animation
- Subtle dot-grid background pattern on surfaces
- Hairline borders, monospaced data tables with alternating row opacity
- Dark-mode native — no light theme

---

## 5. Build & Deploy

### Development

```bash
# Terminal 1: FastAPI backend
cd api && uvicorn main:app --reload --port 8000

# Terminal 2: Vite dev server
cd web && npm run dev    # :5173, proxies /api/* → :8000
```

### Production — Multi-Stage Docker

```dockerfile
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
COPY . .
RUN pip install -e ".[api]"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

FastAPI mounts `web/dist/` via `StaticFiles` and serves `index.html` as the SPA fallback.

### Docker Compose

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
  worker:
    build: .
    command: celery -A engine.worker worker -l info
  redis:
    image: redis:7-alpine
```

Single port (`:8000`) serves both API and web UI.
