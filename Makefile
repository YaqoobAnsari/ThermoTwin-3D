.RECIPEPREFIX := >
.DEFAULT_GOAL := help
.PHONY: help setup lint format test clean

help:  ## Show this help
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the conda env and install the package (editable) + hooks
> conda env create -f env/environment.yml || conda env update -f env/environment.yml
> pip install -e .
> pre-commit install

lint:  ## Ruff lint
> ruff check src tests scripts

format:  ## Ruff format
> ruff format src tests scripts

test:  ## Run the test suite
> pytest -q

clean:  ## Remove caches
> rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
