.PHONY: install dev lint format typecheck test test-cov run evolve prompts skills memory state

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

test:
	pytest tests/

test-cov:
	pytest tests/ --cov --cov-report=term-missing

run:
	python -m src run "$(TASK)"

evolve:
	python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles $(or $(CYCLES),3)

prompts:
	python -m src prompts

skills:
	python -m src skills

memory:
	python -m src memory

state:
	python -m src state
