# Проверки и обслуживание чистового репозитория.
#
# make check   прогнать все проверки; ничего не скачивает, работает на пустом дереве
# make init    поставить крючок перед коммитом
# make fetch   скачать тела корпусов из объектного хранилища и сверить хэши
# make docs    пересобрать оглавление документов
# make strict  проверки в строгом виде: требуются все эталоны и все пути
# make push    протолкнуть ветку в основной удалённый адрес

ROOT   := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PY     ?= python3
BRANCH ?= main

.PHONY: check docs registry leaks release test init fetch strict push help

help:
	@echo "make check   проверки целостности репозитория"
	@echo "make init    поставить крючок перед коммитом"
	@echo "make fetch   скачать тела корпусов и сверить хэши"
	@echo "make docs    пересобрать оглавление документов"
	@echo "make strict  строгие проверки, требуются все эталоны и все пути"
	@echo "make push    протолкнуть ветку $(BRANCH) в основной удалённый адрес"

check: docs registry leaks release
	@echo "проверки пройдены"

docs:
	@$(PY) $(ROOT)tools/build_docs_index.py --check

registry:
	@$(PY) $(ROOT)tools/check_registry.py

leaks:
	@$(PY) $(ROOT)tools/check_no_leaks.py

release:
	@$(PY) $(ROOT)tools/release/check_release.py

test:
	@PYTHONPATH=$(ROOT)src $(PY) -m unittest discover -s $(ROOT)tests -v

strict:
	@$(PY) $(ROOT)tools/build_docs_index.py --check
	@$(PY) $(ROOT)tools/check_registry.py --strict-paths
	@$(PY) $(ROOT)tools/check_no_leaks.py --strict

init:
	@install -m 755 $(ROOT)tools/pre-commit $(ROOT).git/hooks/pre-commit
	@echo "крючок поставлен: .git/hooks/pre-commit"

fetch:
	@bash $(ROOT)data/fetch.sh

# Толкать только после проверок: проверка на тексты закрытых бенчмарков обязана
# отработать до того, как что-то уедет в удалённый адрес.
push: check
	@echo "толкаю ветку $(BRANCH) в основной удалённый адрес"
	@git -C $(ROOT) push origin $(BRANCH)
