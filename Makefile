.PHONY: install data build analysis test lint all

install:            ## install package + dev tools
	pip install -e ".[dev]"

data:               ## download Fingertips indicators, region lookup, council boundaries
	python -m selfharm_model.download

build:              ## time-aligned modelling tables + feature catalogue + boundaries
	python -m selfharm_model.build

analysis:           ## nested + spatial CV, SHAP, GLM, observed/expected, audits, figures (~5 min)
	python -m selfharm_model.analysis

test:
	pytest -q

lint:
	ruff check src tests

all: data build analysis test
