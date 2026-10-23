"""Smoke test: the app loads its bundle, scores customers and renders every tab without error."""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
BUNDLE = APP.parent / "bundle" / "customers.parquet"


@pytest.mark.skipif(not BUNDLE.exists(), reason="no app bundle: run `churn bundle` first")
def test_app_renders_all_tabs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == [
        "Overview", "Risk explorer", "Campaign simulator", "Model card"
    ]  # fmt: skip
    assert at.metric[0].value.endswith("%")  # churn rate tile

    # a stricter budget must not contact more people
    contacted = at.metric[5].value  # "Customers to contact"
    at.slider[5].set_value(0.02).run()  # budget cap
    assert not at.exception, at.exception
    assert at.metric[5].value != contacted and "(2.0%)" in at.metric[5].value
