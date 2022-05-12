SOURCE_GLOB=$(wildcard bin/*.py src/**/*.py tests/**/*.py examples/*.py)

#
IGNORE_PEP=E203,E221,E241,E272,E501,F811

# help scripts to find the right place of wechaty module
export PYTHONPATH=src/


.PHONY: lint
lint: pylint pycodestyle flake8 mypy


# disable: TODO list temporay
.PHONY: pylint
pylint:
	pylint \
		--load-plugins pylint_quotes \
		--disable=W0511,R0801,cyclic-import,C4001 \
		$(SOURCE_GLOB)

.PHONY: install
install:
	pip3 install -r requirements.txt
	pip3 install -r requirements-dev.txt

.PHONY: pytest
pytest:
	pytest src/ tests/

.PHONY: test-unit
test-unit: pytest

.PHONY: test
test: pytest pylint

.PHONY: dist
dist:
	python3 setup.py sdist bdist_wheel

.PHONY: publish
publish:
	PATH=~/.local/bin:${PATH} twine upload dist/*

.PHONY: run
run:
	python serving.py