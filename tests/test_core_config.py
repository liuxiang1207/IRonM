"""运行配置（spec [2026-10-20] §2）。"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ironm.core.config import ConfigError, load_settings


def test_defaults() -> None:
    s = load_settings({})
    assert s.env == "dev"
    assert s.log_level == "info"
    assert s.db_path == Path("data/ironm.sqlite")


def test_db_path_configurable() -> None:
    assert load_settings({"IRONM_DB_PATH": "/tmp/x.sqlite"}).db_path == Path("/tmp/x.sqlite")


def test_reads_prefixed_vars() -> None:
    s = load_settings({"IRONM_ENV": "prod", "IRONM_LOG_LEVEL": "warning"})
    assert s.env == "prod"
    assert s.log_level == "warning"


def test_ignores_unprefixed_vars() -> None:
    assert load_settings({"PATH": "/bin", "HOME": "/x"}).env == "dev"


@pytest.mark.parametrize(
    ("environ", "var"),
    [
        ({"IRONM_ENV": "staging"}, "IRONM_ENV"),
        ({"IRONM_LOG_LEVEL": "loud"}, "IRONM_LOG_LEVEL"),
        ({"IRONM_FOO": "1"}, "IRONM_FOO"),
    ],
)
def test_invalid_or_unknown_fails_naming_var(environ: dict[str, str], var: str) -> None:
    with pytest.raises(ConfigError, match=var):
        load_settings(environ)


def test_settings_immutable() -> None:
    s = load_settings({})
    with pytest.raises(ValueError):
        s.env = "prod"


def _start_app(environ: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRONM_")} | environ
    return subprocess.run(
        [sys.executable, "-c", "import ironm.api.app"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_app_startup_fails_on_bad_config() -> None:
    result = _start_app({"IRONM_FOO": "1"})
    assert result.returncode != 0
    assert "IRONM_FOO" in result.stderr


def test_app_starts_with_default_config() -> None:
    result = _start_app({})
    assert result.returncode == 0, result.stderr
