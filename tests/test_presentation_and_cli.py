from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from hybrid_power.cli import EXIT_ERROR, EXIT_OK, CommandLineInterface
from hybrid_power.config import FleetConfig, ProfileConfig, ScenarioConfig
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.fleet import run_fleet_study
from hybrid_power.presentation import ChartFactory
from hybrid_power.presentation.charts import HEATMAP_METRICS
from hybrid_power.service import SimulationOutcome, SimulationService


@pytest.fixture(scope="module")
def outcome() -> SimulationOutcome:
    return SimulationService().run(ScenarioConfig(profile=ProfileConfig(days=3)))


class TestChartFactory:
    def test_every_chart_builds(self, outcome: SimulationOutcome) -> None:
        charts = ChartFactory(currency="UGX")
        metrics = list(outcome.report.metrics.values())
        optimizer = outcome.results["optimizer"]
        figures = [
            charts.profile_overview(outcome.profile, solar_rated_kw=8),
            charts.power_flow(optimizer),
            charts.daily_energy(optimizer),
            charts.battery_soc(outcome.results),
            charts.cumulative_cost(outcome.results),
            charts.cost_breakdown(metrics),
            charts.energy_mix(metrics),
            charts.environmental_impact(metrics),
        ]
        for figure in figures:
            assert figure.data, "figure has no traces"
            assert figure.layout.title.text

    def test_power_flow_has_one_trace_per_path(self, outcome: SimulationOutcome) -> None:
        figure = ChartFactory().power_flow(outcome.results["optimizer"])
        names = [trace.name for trace in figure.data]
        assert names == ["Solar", "Battery", "Grid", "Generator", "Unmet", "Demand"]

    @pytest.mark.parametrize("metric", list(HEATMAP_METRICS))
    def test_heatmaps_for_every_metric(self, outcome: SimulationOutcome, metric: str) -> None:
        charts = ChartFactory()
        result = outcome.results["optimizer"]
        hourly = charts.hourly_heatmap(result, metric)
        z = hourly.data[0].z
        assert len(z) == 3  # days
        assert len(z[0]) == 24  # hours
        calendar = charts.calendar_heatmap(result, metric)
        cells = [value for row in calendar.data[0].z for value in row if value == value]
        assert len(cells) == 3

    def test_hourly_heatmap_values_match_records(self, outcome: SimulationOutcome) -> None:
        result = outcome.results["baseline"]
        figure = ChartFactory().hourly_heatmap(result, "Operating cost")
        first_day_total = sum(r.total_cost for r in result.records[:24])
        assert sum(figure.data[0].z[0]) == pytest.approx(first_day_total)
        hover = figure.data[0].hovertemplate
        assert "Demand" in hover
        assert "Cost" in hover

    def test_calendar_layout(self, outcome: SimulationOutcome) -> None:
        figure = ChartFactory().calendar_heatmap(outcome.results["optimizer"])
        heatmap = figure.data[0]
        assert list(heatmap.x) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        # 1 Oct 2026 is a Thursday; the 3-day profile spans one calendar week.
        assert heatmap.text[0][3] == "1"
        assert heatmap.text[0][0] == ""

    def test_savings_calendar_is_centred(self, outcome: SimulationOutcome) -> None:
        figure = ChartFactory().savings_calendar(
            outcome.results["baseline"], outcome.results["optimizer"]
        )
        heatmap = figure.data[0]
        assert heatmap.zmid == 0
        saved = [v for row in heatmap.z for v in row if v == v]
        assert sum(saved) == pytest.approx(outcome.report.comparison("optimizer").cost_savings)

    def test_unknown_heatmap_metric(self, outcome: SimulationOutcome) -> None:
        with pytest.raises(ConfigurationError, match="Unknown heatmap metric"):
            ChartFactory().hourly_heatmap(outcome.results["optimizer"], "Happiness")

    def test_fleet_charts(self) -> None:
        study = run_fleet_study(
            ScenarioConfig(profile=ProfileConfig(days=2)), FleetConfig(sites=4, fleet_sizes=(1, 4))
        )
        charts = ChartFactory()
        convergence = charts.fleet_convergence(study.report)
        assert list(convergence.data[0].x) == ["1", "4"]
        distribution = charts.fleet_distribution(study.fleet)
        assert [trace.name for trace in distribution.data] == ["solar_first", "optimizer"]
        drivers = charts.fleet_drivers(study.fleet)
        assert len(drivers.data[0].x) == 4

    def test_metrics_table(self, outcome: SimulationOutcome) -> None:
        table = ChartFactory.metrics_table(list(outcome.report.metrics.values()))
        assert list(table.columns) == ["baseline", "solar_first", "optimizer"]
        assert "Diesel used (L)" in table.index


class TestCommandLine:
    def run(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = CommandLineInterface(stdout=out, stderr=err).run(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_simulate_prints_comparison(self) -> None:
        code, out, _ = self.run("simulate", "--days", "2", "--seed", "3")
        assert code == EXIT_OK
        assert "optimizer: saves" in out
        assert "Recommended strategy" in out

    def test_simulate_selected_strategy_and_export(self, tmp_path: Path) -> None:
        code, out, _ = self.run(
            "simulate", "--days", "2", "--strategy", "optimizer", "--export-dir", str(tmp_path)
        )
        assert code == EXIT_OK
        assert "solar_first" not in out
        exported = {path.name for path in tmp_path.iterdir()}
        assert exported == {
            "metrics.csv",
            "profile.csv",
            "scenario.json",
            "results_baseline.csv",
            "results_optimizer.csv",
        }

    def test_generate_profile_then_simulate_from_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "profile.csv"
        code, out, _ = self.run("generate-profile", "--days", "2", "--output", str(csv))
        assert code == EXIT_OK
        assert "48 steps" in out
        code, out, _ = self.run("simulate", "--profile", str(csv))
        assert code == EXIT_OK
        assert "48 steps" in out

    def test_config_file(self, tmp_path: Path) -> None:
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps({"name": "Test site", "profile": {"days": 1}}))
        code, out, _ = self.run("simulate", "--config", str(path))
        assert code == EXIT_OK
        assert "Test site" in out

    def test_fleet_command(self, tmp_path: Path) -> None:
        code, out, _ = self.run(
            "fleet", "--days", "2", "--sizes", "1", "3", "--export-dir", str(tmp_path)
        )
        assert code == EXIT_OK
        assert "Simulating 3 base stations" in out
        assert "Direction is" in out
        assert {p.name for p in tmp_path.iterdir()} == {"fleet_sites.csv", "fleet_consistency.csv"}

    def test_fleet_command_default_sizes(self) -> None:
        code, out, _ = self.run("fleet", "--days", "1", "--sites", "6")
        assert code == EXIT_OK
        assert "Simulating 6 base stations" in out

    def test_errors_are_reported_cleanly(self, tmp_path: Path) -> None:
        code, _, err = self.run("simulate", "--days", "0")
        assert code == EXIT_ERROR
        assert err.startswith("error: days")
        code, _, err = self.run("simulate", "--profile", str(tmp_path / "missing.csv"))
        assert code == EXIT_ERROR
        assert "not found" in err

    def test_unknown_strategy_is_rejected_by_parser(self) -> None:
        with pytest.raises(SystemExit):
            self.run("simulate", "--strategy", "magic")
