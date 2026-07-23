from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_in_clean_interpreter(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_package_import_is_logging_passive() -> None:
    result = _run_in_clean_interpreter(
        """
        import logging

        root_logger = logging.getLogger()
        package_logger = logging.getLogger("smolllm")
        root_handler = logging.StreamHandler()
        package_handler = logging.StreamHandler()
        root_logger.handlers = [root_handler]
        root_logger.setLevel(logging.ERROR)
        package_logger.handlers = [package_handler]
        package_logger.setLevel(logging.CRITICAL)
        package_logger.propagate = True

        from smolllm import ask_llm

        assert callable(ask_llm)
        assert root_logger.handlers == [root_handler]
        assert root_logger.level == logging.ERROR
        assert package_logger.handlers == [package_handler]
        assert package_logger.level == logging.CRITICAL
        assert package_logger.propagate is True
        """
    )

    assert result.returncode == 0, result.stderr


def test_ask_llm_import_does_not_load_embeddings_or_rich_display() -> None:
    result = _run_in_clean_interpreter(
        """
        import sys

        from smolllm import ask_llm

        assert callable(ask_llm)
        assert "smolllm.embeddings" not in sys.modules
        assert "smolllm.display" not in sys.modules
        assert not any(name == "rich" or name.startswith("rich.") for name in sys.modules)
        """
    )

    assert result.returncode == 0, result.stderr
