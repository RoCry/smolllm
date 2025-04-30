#!/usr/bin/env python3
"""
Script to bump version numbers.
Usage: python -m tools.bump_version [major|minor|patch]
"""

import re
import sys
from pathlib import Path


def read_version(file_path: Path) -> str:
    """Read version from a Python file"""
    content = file_path.read_text()
    if file_path.suffix == ".py":
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    elif file_path.suffix == ".toml":
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError(f"Version not found in {file_path}")
    return match.group(1)


def write_version(file_path: Path, new_version: str) -> None:
    """Write version to a file"""
    content = file_path.read_text()
    if file_path.suffix == ".py":
        new_content = re.sub(
            r'__version__\s*=\s*["\']([^"\']+)["\']',
            f'__version__ = "{new_version}"',
            content,
        )
    elif file_path.suffix == ".toml":
        new_content = re.sub(r'version\s*=\s*["\']([^"\']+)["\']', f'version = "{new_version}"', content)
    file_path.write_text(new_content)


def bump_version(version: str, bump_type: str) -> str:
    """Bump version number"""
    major, minor, patch = map(int, version.split("."))

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError("Invalid bump type. Use 'major', 'minor', or 'patch'")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print("Usage: python -m tools.bump_version [major|minor|patch]")
        sys.exit(1)

    bump_type = sys.argv[1]
    root_dir = Path(__file__).parent.parent

    # Files to update
    files = [root_dir / "src" / "smolllm" / "__init__.py", root_dir / "pyproject.toml"]

    # Read current version from __init__.py
    current_version = read_version(files[0])
    new_version = bump_version(current_version, bump_type)

    # Update all files
    for file_path in files:
        write_version(file_path, new_version)
        print(f"Updated {file_path}")

    print(f"Version bumped: {current_version} -> {new_version}")


if __name__ == "__main__":
    main()
