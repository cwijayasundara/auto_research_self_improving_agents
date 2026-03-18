.PHONY: install dev lint format typecheck test test-cov run evolve prompts skills memory state

PYTHON := .venv/bin/python
PIP := .venv/bin/pip

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e .

dev: venv
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest tests/

test-cov:
	$(PYTHON) -m pytest tests/ --cov --cov-report=term-missing

run:
	$(PYTHON) -m src run "$(TASK)"

evolve:
	$(PYTHON) -m src evolve --tasks-file tasks/research_tasks.json --max-cycles $(or $(CYCLES),3)

prompts:
	$(PYTHON) -m src prompts

skills:
	$(PYTHON) -m src skills

memory:
	$(PYTHON) -m src memory

state:
	$(PYTHON) -m src state
