.PHONY: clean build install-dev test-release manual-release bump-patch bump-minor bump-major

UV_RUN ?= uv run
SMOLLLM_ENV_FILE ?= $(HOME)/.env.smolllm
SMOLLLM_RUN = $(UV_RUN) --env-file $(SMOLLLM_ENV_FILE)

# Helper functions
define get_version
$(shell $(UV_RUN) python -c "from src.smolllm import __version__; print(__version__)")
endef

VERSION := $(call get_version)

clean:
	rm -rf dist/ build/ *.egg-info/

install-dev: clean
	uv sync --all-extras --dev

build: install-dev
	$(UV_RUN) python -m build

test:
	$(UV_RUN) pytest -s -v tests/*

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
define do_version_bump
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: Could not determine current version"; \
		exit 1; \
	fi
	@if [ ! -f "$(SMOLLLM_ENV_FILE)" ]; then \
		echo "Error: SMOLLLM_ENV_FILE not found: $(SMOLLLM_ENV_FILE)"; \
		echo "Set SMOLLLM_ENV_FILE=/path/to/.env or create $(SMOLLLM_ENV_FILE)"; \
		exit 1; \
	fi
	@echo "Current version: $(VERSION)"
	@read -p "Are you sure you want to bump $(1) version? [y/N] " confirm && [ $$confirm = "y" ]
	$(UV_RUN) tools/bump_version.py $(1)
	@NEW_VERSION=$$($(UV_RUN) python -c "from src.smolllm import __version__; print(__version__)") && \
	if [ -z "$$NEW_VERSION" ]; then \
		echo "Error: Could not determine new version"; \
		exit 1; \
	fi && \
	echo "Version bumped: $(VERSION) -> $$NEW_VERSION" && \
	$(SMOLLLM_RUN) examples/simple.py && echo "\nTest passed\n" && \
	read -p "Do you want to commit, tag and push? [y/N] " confirm && [ $$confirm = "y" ] && \
	git commit -am "chore: bump version to $$NEW_VERSION" && \
	git tag -m "Release v$$NEW_VERSION" v$$NEW_VERSION && \
	git push && git push --tags
endef

bump-patch:
	$(call do_version_bump,patch)

bump-minor:
	$(call do_version_bump,minor)

bump-major:
	$(call do_version_bump,major)

# Development commands
update-providers:
	$(UV_RUN) tools/update_providers.py
