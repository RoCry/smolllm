.PHONY: clean build test-release release install-dev

VERSION := $(shell python -c "from src.smolllm import __version__; print(__version__)")

clean:
	rm -rf dist/ build/ *.egg-info/

install-dev: clean
	uv pip install -e ".[dev]"

# Make build depend on install-dev to ensure build tools are available
build: install-dev
	python -m build

test-release: build
	@echo "Uploading version $(VERSION) to Test PyPI..."
	twine upload --repository testpypi dist/*
	@echo "Testing installation from Test PyPI..."
	uv pip install --index-url https://test.pypi.org/simple/ --no-deps smolllm==$(VERSION)
	@echo "Test installation completed. Please verify the package works correctly."

release: build
	@echo "Uploading version $(VERSION) to PyPI..."
	twine upload dist/*
	@echo "Release completed! Version $(VERSION) is now available on PyPI."

# Development commands
update-providers:
	python tools/update_providers.py