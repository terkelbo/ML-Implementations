venv_dir = .venv

all: dev-setup

dev-setup: $(venv_dir)/.made


$(venv_dir)/.made:
	python3 -m venv $(venv_dir)
	$(venv_dir)/bin/pip install --upgrade pip
	$(venv_dir)/bin/pip install poetry
	$(venv_dir)/bin/poetry install -vvv --no-root
	$(venv_dir)/bin/pre-commit install
