STUDY_VENV ?= /mnt/d/dev/codex/vdt-dynamic-span/.venv-study
PYTHON ?= $(STUDY_VENV)/bin/python

.PHONY: study-env test-core test-all reference-canary

study-env:
	@mkdir -p $(dir $(STUDY_VENV))
	@if [ ! -x "$(STUDY_VENV)/bin/python" ]; then uv venv "$(STUDY_VENV)" --python 3.10; fi
	uv pip install --python "$(STUDY_VENV)/bin/python" -r requirements-study.txt

test-core:
	$(PYTHON) -m pytest tests/core

test-all:
	$(PYTHON) -m pytest

reference-canary:
	PYTHONPATH=. $(PYTHON) scripts/vdt/reference_canary.py --output artifacts/reference_canary.json
