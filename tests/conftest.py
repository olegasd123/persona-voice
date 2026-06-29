from __future__ import annotations

import os
from pathlib import Path

import pytest

from personavoice.server.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(autouse=True)
def restore_env_after_test() -> None:
    """Keep `.env` loads in CLI/settings tests from leaking into later tests."""
    env = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env)


@pytest.fixture
def config_dir() -> Path:
    return CONFIG_DIR


def make_settings(backend: str) -> Settings:
    return Settings(
        backend=backend,
        config_dir=CONFIG_DIR,
        models_dir=REPO_ROOT / "models",
    )


@pytest.fixture(params=["mac", "cuda"])
def settings(request: pytest.FixtureRequest) -> Settings:
    return make_settings(request.param)
