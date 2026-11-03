PY ?= python3
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: check test weights init help
help:
	@echo 'make check   source, configuration and publication checks'
	@echo 'make test    CPU tests on synthetic inputs'
	@echo 'make weights verify materialized Git LFS weights'
	@echo 'make init    install the pre-commit check'

check:
	@$(PY) tools/release/check_release.py
	@$(PY) tools/check_no_leaks.py

test:
	@PYTHONPATH=$(ROOT)src PYTHONDONTWRITEBYTECODE=1 $(PY) -m pytest tests -q

weights:
	@$(PY) tools/release/check_release.py --weights

init:
	@install -m 755 tools/pre-commit "$$(git rev-parse --git-path hooks)/pre-commit"
