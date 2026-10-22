"""Load and render the SQL templates in sql/. Templates use ${name} placeholders."""

from __future__ import annotations

from string import Template

from churn.config import Settings


def render(name: str, settings: Settings, **params: object) -> str:
    text = (settings.sql_dir / name).read_text(encoding="utf-8")
    return Template(text).substitute({k: str(v) for k, v in params.items()})
