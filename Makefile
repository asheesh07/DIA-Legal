.PHONY: install dev build test docker-build docker-run clean help

PYTHON  := python3
PIP     := pip3
PORT    := 8000
IMAGE   := dia-legal

help:
	@echo ""
	@echo "  DIA-Legal — available commands"
	@echo ""
	@echo "  make install        Install Python + Node dependencies"
	@echo "  make dev            Start backend (port 8000) + frontend (port 5173)"
	@echo "  make backend        Backend only"
	@echo "  make frontend       Frontend dev server only"
	@echo "  make build          Build frontend for production"
	@echo "  make test           Run full demo test suite (server must be running)"
	@echo "  make eval           Run RAG evaluation suite"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo "  make clean          Remove __pycache__, .venv leftovers, build artifacts"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────

install:
	$(PIP) install -r requirements.txt
	cd frontend && npm install

# ── Development ────────────────────────────────────────────────────────────

backend:
	uvicorn main:app --port $(PORT) --reload

frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting backend on :$(PORT) and frontend on :5173"
	@(uvicorn main:app --port $(PORT) --reload &) && cd frontend && npm run dev

# ── Build ──────────────────────────────────────────────────────────────────

build:
	cd frontend && npm run build
	@echo "Frontend built → frontend/dist/"
	@echo "Serve with: uvicorn main:app --port $(PORT)"

# ── Test ───────────────────────────────────────────────────────────────────

test:
	$(PYTHON) demo_full_test.py

eval:
	$(PYTHON) eval_runner.py

# ── Docker ─────────────────────────────────────────────────────────────────

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):$(PORT) \
		--env-file .env \
		-v $$(pwd)/data:/app/data \
		$(IMAGE)

# ── Clean ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -not -path "./.git/*" | xargs rm -rf
	find . -name "*.pyc" -delete
	rm -rf frontend/dist frontend/node_modules/.vite
	@echo "Cleaned."
