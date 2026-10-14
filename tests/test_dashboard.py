"""Headless tests of the Streamlit dashboard using Streamlit's AppTest harness."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_dashboard_renders_without_errors(app: AppTest) -> None:
    assert not app.exception
    assert app.title[0].value == "Hybrid Power Optimiser"
    labels = [metric.label for metric in app.metric]
    assert "Operating cost" in labels
    assert "Diesel used" in labels
    assert any("Recommended strategy" in info.value for info in app.info)


def test_changing_inputs_reruns_simulation(app: AppTest) -> None:
    app.slider(key="default:days").set_value(5).run()
    assert not app.exception
    steps = next(metric for metric in app.metric if metric.label == "Steps")
    assert steps.value == "120"


def test_depleted_battery_start_is_accepted(app: AppTest) -> None:
    app.slider(key="default:win").set_value((0.5, 0.55)).run()
    app.slider(key="default:init").set_value(0.1).run()
    # An initial SoC below min_soc models a depleted battery and is a valid scenario.
    assert not app.exception
    assert not app.error
    assert app.metric


def test_heatmap_tab_switches_metric(app: AppTest) -> None:
    app.selectbox(key="heat_metric").set_value("Generator output (kWh)").run()
    assert not app.exception


def test_fleet_study_runs_from_form(app: AppTest) -> None:
    next(s for s in app.slider if s.label == "Base stations (largest fleet)").set_value(5)
    next(s for s in app.slider if s.label == "Days simulated per site").set_value(7)
    next(b for b in app.button if b.label == "Run fleet study").click().run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Mean cost saving" in labels
    sites = next(metric for metric in app.metric if metric.label == "Sites simulated")
    assert sites.value == "5"
    alerts = [*app.success, *app.warning, *app.error]
    assert any("Direction is" in alert.value for alert in alerts)
