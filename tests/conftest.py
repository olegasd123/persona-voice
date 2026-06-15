from __future__ import annotations

from pathlib import Path

import pytest

from personavoice.server.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


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
