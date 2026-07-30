set default-list

export UV_RUN := env_var_or_default("UV_RUN", "uv run")
export SMOLLLM_ENV_FILE := env_var_or_default("SMOLLLM_ENV_FILE", home_directory() + "/.env.smolllm")

# Remove build artifacts.
clean:
    rm -rf dist/ build/ *.egg-info/

# Install all development dependencies.
install-dev: clean
    uv sync --all-extras --dev

# Build the package.
build: install-dev
    {{ UV_RUN }} python -m build

# Run the test suite.
test:
    {{ UV_RUN }} pytest -s -v tests/*

# Upload to and install from Test PyPI.
test-release: build
    ./scripts/release.sh test

# Upload to PyPI manually.
manual-release: build
    ./scripts/release.sh manual

# Bump the patch version, smoke-test, commit, tag, and push.
bump-patch:
    ./scripts/release.sh bump patch

# Bump the minor version, smoke-test, commit, tag, and push.
bump-minor:
    ./scripts/release.sh bump minor

# Bump the major version, smoke-test, commit, tag, and push.
bump-major:
    ./scripts/release.sh bump major

# Update provider configurations.
update-providers:
    {{ UV_RUN }} tools/update_providers.py
