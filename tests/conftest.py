"""Shared pytest setup.

Redirects temp files into `<repo>/.pytest_tmp_runs/` — works around
permission errors with the default Windows temp directory.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

_RUN_TMP = (
    Path(__file__).resolve().parents[1]
    / ".pytest_tmp_runs"
    / f"run-{os.getpid()}-{uuid.uuid4().hex}"
)


def pytest_configure() -> None:
    _RUN_TMP.mkdir(parents=True, exist_ok=True)

    os.environ["TMP"] = str(_RUN_TMP)
    os.environ["TEMP"] = str(_RUN_TMP)
    tempfile.tempdir = str(_RUN_TMP)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in request.node.name
    )[:60]
    path = _RUN_TMP / "tmp_path" / f"{safe_name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
