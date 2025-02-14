.PHONY: clean build install-dev test-release release bump-patch bump-minor bump-major

VERSION := $(shell python -c "from src.smolllm import __version__; print(__version__)")

clean:
	rm -rf dist/ build/ *.egg-info/

install-dev: clean
	uv pip install -e ".[dev]"

build: install-dev
	python -m build

# Manual release commands
test-release: build
	@echo "Uploading version $(VERSION) to Test PyPI..."
	twine upload --repository testpypi dist/*
	@echo "Testing installation from Test PyPI..."
	uv pip install --index-url https://test.pypi.org/simple/ --no-deps smolllm==$(VERSION)
	@echo "Test installation completed. Please verify the package works correctly."

# Manual release (for emergency or when GitHub Actions is not available)
manual-release: build
	@echo "⚠️  Warning: Manual release should only be used when GitHub Actions release is not possible"
	@echo "Current version: $(VERSION)"
	@read -p "Are you sure you want to release manually? [y/N] " confirm && [ $$confirm = "y" ]
	@echo "Uploading version $(VERSION) to PyPI..."
	twine upload dist/*
	@echo "Release completed! Version $(VERSION) is now available on PyPI."

# Version management
bump-patch:
	python tools/bump_version.py patch
	@echo "Don't forget to:"
	@echo "1. git commit -am 'chore: bump version to $(VERSION)'"
	@echo "2. git tag v$(VERSION)"
	@echo "3. git push && git push --tags"

bump-minor:
	python tools/bump_version.py minor
	@echo "Don't forget to:"
	@echo "1. git commit -am 'chore: bump version to $(VERSION)'"
	@echo "2. git tag v$(VERSION)"
	@echo "3. git push && git push --tags"

bump-major:
	python tools/bump_version.py major
	@echo "Don't forget to:"
	@echo "1. git commit -am 'chore: bump version to $(VERSION)'"
	@echo "2. git tag v$(VERSION)"
	@echo "3. git push && git push --tags"

# Development commands
update-providers:
	python tools/update_providers.py