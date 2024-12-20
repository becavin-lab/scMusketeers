.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
USING_POETRY=$(shell grep "tool.poetry" pyproject.toml && echo "yes")
VERSION=$(shell poetry version | awk '{print $$2}')

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: show
show:             ## Show the current environment.
	@echo "Current environment:"
	$(ENV_PREFIX)poetry env info

.PHONY: install
install:          ## Install the project in dev mode.
	@echo "Run scMusketeers install - create poetry virtual env"
	$(ENV_PREFIX)poetry install

.PHONY: install-dev
install-dev:          ## Install the project in dev mode.
	@echo "Run scMusketeers install - create poetry virtual env for dev"
	$(ENV_PREFIX)poetry install --with dev

.PHONY: fmt
fmt:              ## Format code using black & isort.
	@echo "Run scMusketeers file formatting"
	$(ENV_PREFIX)poetry run isort scmusketeers/ tests/
	$(ENV_PREFIX)poetry run black -l 79 scmusketeers/ tests/
	$(ENV_PREFIX)no_implicit_optional scmusketeers/

.PHONY: lint
lint:             ## Run pep8, black, mypy linters.
	@echo "Run scMusketeers linting"
	$(ENV_PREFIX)poetry run flake8 scmusketeers/
	$(ENV_PREFIX)poetry run black -l 100 --check scmusketeers/ tests/
	$(ENV_PREFIX)poetry run mypy --ignore-missing-imports scmusketeers/ tests/

.PHONY: test
test:             ## Run tests and generate coverage report.
	$(ENV_PREFIX)poetry run pytest -v --cov-config .coveragerc --cov=scmusketeers -l --tb=short --maxfail=1 tests/
	$(ENV_PREFIX)poetry run coverage xml
	$(ENV_PREFIX)poetry run coverage html

.PHONY: watch
watch:            ## Run tests on every change.
	ls **/**.py | $(ENV_PREFIX)pytest -s -vvv -l --tb=long --maxfail=1 tests/

.PHONY: docs
docs:             ## Build the documentation.
	@echo "Building documentation ..."
	@$(ENV_PREFIX)poetry run mkdocs build
##	URL="site/index.html"; open $$URL || xdg-open $$URL || sensible-browser $$URL || x-www-browser $$URL || gnome-open $$URL

.PHONY: ci
ci:          ## Run a continuous integration : Add every change to git and create a new tag for continuous integration.
	@echo "WARNING: You need first to add changes to git with git add"
	@echo "Push change to github and run continous integration scripts"
	@git add --all
	@git commit -m "Continuous integration 🔄 tests-$(VERSION)"
	@echo "creating git tag : tests-$(VERSION)"
	@git tag tests-$(VERSION)-1
	@git push -u origin HEAD --tags
	@echo "Github Actions will detect the new tag and run the continuous integration process."

.PHONY: release
release:          ## Create a new tag for release.
	@echo "WARNING: This operation will create a version tag and push to github"
	@echo "Reading version $(VERSION) from: pyproject.toml"
	@echo "Saving version to: scmusketeers/tools/VERSION"
	@echo "${VERSION}" > "scmusketeers/tools/VERSION"
	@$(ENV_PREFIX)poetry run gitchangelog > HISTORY.md
	@git add HISTORY.md pyproject.toml scmusketeers/tools/VERSION
	@git commit -m "release: version $(VERSION) 🚀"
	@echo "creating git tag : release-$(VERSION)"
	@git tag release-$(VERSION)
	@git push -u origin HEAD --tags
	@echo "Github Actions will detect the new tag and release the new version."



.PHONY: clean
clean:            ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycachemt__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf build
	@rm -rf dist
	@rm -rf site
	@rm -rf *.egg-info
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build

