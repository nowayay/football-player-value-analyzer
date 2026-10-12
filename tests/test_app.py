"""Smoke test: the Streamlit app loads from committed files without errors."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_loads_with_four_tabs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Undervalued", "Overvalued", "Player lookup", "Model performance"]
    assert len(at.dataframe) >= 2  # ranking tables are rendered


def test_player_lookup_search_runs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].input("Saka").run()
    assert not at.exception
    assert any("Market value" in m.label for m in at.metric)
