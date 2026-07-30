#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly UV_RUN_VALUE="${UV_RUN-uv run}"

cd "$REPO_ROOT"

if [[ -z "${UV_RUN_VALUE//[[:space:]]/}" ]]; then
    echo "Error: UV_RUN must not be empty" >&2
    exit 1
fi

run_uv() {
    "$BASH" -c "$UV_RUN_VALUE \"\$@\"" -- "$@"
}

current_version() {
    local version
    version="$(run_uv python -c "from src.smolllm import __version__; print(__version__)")"
    if [[ -z "$version" ]]; then
        echo "Error: Could not determine current version" >&2
        exit 1
    fi
    printf '%s\n' "$version"
}

confirm() {
    local prompt="$1"
    local answer
    read -r -p "$prompt [y/N] " answer
    [[ "$answer" == "y" ]]
}

test_release() {
    local version
    version="$(current_version)"

    echo "Uploading version $version to Test PyPI..."
    twine upload --repository testpypi dist/*
    echo "Testing installation from Test PyPI..."
    uv pip install --index-url https://test.pypi.org/simple/ --no-deps "smolllm==$version"
    echo "Test installation completed. Please verify the package works correctly."
}

manual_release() {
    local version
    version="$(current_version)"

    echo "⚠️  Warning: Manual release should only be used when GitHub Actions release is not possible"
    echo "Current version: $version"
    if ! confirm "Are you sure you want to release manually?"; then
        echo "Release aborted." >&2
        exit 1
    fi
    echo "Uploading version $version to PyPI..."
    twine upload dist/*
    echo "Release completed! Version $version is now available on PyPI."
}

bump_version() {
    local bump_type="$1"
    local env_file="${SMOLLLM_ENV_FILE-$HOME/.env.smolllm}"
    local old_version
    local new_version

    case "$bump_type" in
        patch | minor | major) ;;
        *)
            echo "Error: bump type must be patch, minor, or major" >&2
            exit 2
            ;;
    esac

    old_version="$(current_version)"
    if [[ ! -f "$env_file" ]]; then
        echo "Error: SMOLLLM_ENV_FILE not found: $env_file" >&2
        echo "Set SMOLLLM_ENV_FILE=/path/to/.env or create $env_file" >&2
        exit 1
    fi

    echo "Current version: $old_version"
    if ! confirm "Are you sure you want to bump $bump_type version?"; then
        echo "Version bump aborted." >&2
        exit 1
    fi

    run_uv tools/bump_version.py "$bump_type"
    new_version="$(current_version)"
    echo "Version bumped: $old_version -> $new_version"
    run_uv --env-file "$env_file" examples/simple.py
    printf '\nTest passed\n\n'

    if ! confirm "Do you want to commit, tag and push?"; then
        echo "Commit, tag, and push aborted." >&2
        exit 1
    fi

    git commit -am "chore: bump version to $new_version"
    git tag -m "Release v$new_version" "v$new_version"
    git push
    git push --tags
}

usage() {
    echo "Usage: $0 {test|manual|bump <patch|minor|major>}" >&2
}

case "${1:-}" in
    test)
        [[ $# -eq 1 ]] || {
            usage
            exit 2
        }
        test_release
        ;;
    manual)
        [[ $# -eq 1 ]] || {
            usage
            exit 2
        }
        manual_release
        ;;
    bump)
        [[ $# -eq 2 ]] || {
            usage
            exit 2
        }
        bump_version "$2"
        ;;
    *)
        usage
        exit 2
        ;;
esac
