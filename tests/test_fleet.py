from __future__ import annotations

import pytest

from hybrid_power.analytics.metrics import PerformanceMetrics
from hybrid_power.config import FleetConfig, ProfileConfig, ScenarioConfig
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.fleet import (
    ConsistencyReport,
    ConsistencyStudy,
    FleetGenerator,
    FleetResult,
    FleetSimulator,
    FleetSizeSummary,
    SiteOutcome,
    SummaryStatistics,
    run_fleet_study,
)
from hybrid_power.fleet.consistency import describe_fleets

TEMPLATE = ScenarioConfig(profile=ProfileConfig(days=2))


@pytest.fixture(scope="module")
def fleet_study() -> tuple[FleetResult, ConsistencyReport]:
    outcome = run_fleet_study(TEMPLATE, FleetConfig(sites=8, fleet_sizes=(1, 4, 8)))
    return outcome.fleet, outcome.report


def metrics(name: str, cost: float, fuel: float = 10.0, unmet: float = 0.0) -> PerformanceMetrics:
    return PerformanceMetrics(
        strategy_name=name, currency="USD", total_cost=cost, grid_cost=cost, generator_cost=0,
        battery_cost=0, energy_demand_kwh=100, energy_served_kwh=100 - unmet,
        unmet_energy_kwh=unmet, unmet_hours=int(unmet > 0), solar_used_kwh=0,
        solar_curtailed_kwh=0, grid_energy_kwh=100, generator_energy_kwh=0,
        generator_dumped_kwh=0, battery_throughput_kwh=0, fuel_used_l=fuel, generator_hours=0,
        generator_starts=0, emissions_kg=0, average_soc=0.5, minimum_soc=0.5,
    )  # fmt: skip


def site(site_id: str, base: float, cand: float, cand_unmet: float = 0.0) -> SiteOutcome:
    return SiteOutcome(
        site_id=site_id,
        load_scale=1.0,
        grid_availability=0.9,
        outage_hours=10,
        average_load_kw=3.0,
        metrics={
            "baseline": metrics("baseline", base, fuel=20),
            "optimizer": metrics("optimizer", cand, fuel=5, unmet=cand_unmet),
        },
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class TestFleetConfig:
    def test_defaults_and_json_round_trip(self) -> None:
        config = FleetConfig()
        assert config.fleet_sizes[-1] == config.sites == 50
        assert FleetConfig.from_json(config.to_json()) == config

    def test_with_sites_builds_size_ladder(self) -> None:
        assert FleetConfig.with_sites(30).fleet_sizes == (1, 5, 10, 25, 30)
        assert FleetConfig.with_sites(5).fleet_sizes == (1, 5)
        assert FleetConfig.with_sites(1).fleet_sizes == (1,)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sites": 0},
            {"sites": 10, "fleet_sizes": ()},
            {"sites": 10, "fleet_sizes": (5, 1)},
            {"sites": 10, "fleet_sizes": (5, 20)},
            {"load_scale": (1.5, 0.5)},
            {"outages_per_day": (0.0,)},
            {"solar_mean_clearness": (0.0, 2.0)},
        ],
    )
    def test_invalid(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ConfigurationError):
            FleetConfig(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class TestFleetGenerator:
    def test_sites_vary_within_ranges(self) -> None:
        fleet = FleetConfig(sites=20, fleet_sizes=(20,), load_scale=(0.5, 1.5))
        sites = FleetGenerator(TEMPLATE, fleet).generate()
        assert len(sites) == 20
        assert len({s.site_id for s in sites}) == 20
        assert len({s.config.profile.seed for s in sites}) == 20
        for spec in sites:
            profile = spec.config.profile
            assert 0.5 <= spec.load_scale <= 1.5
            assert profile.peak_load_kw == pytest.approx(
                TEMPLATE.profile.peak_load_kw * spec.load_scale
            )
            assert 0.2 <= profile.outages_per_day <= 3.0
            # Equipment is shared by every site.
            assert spec.config.battery == TEMPLATE.battery
            assert spec.config.generator == TEMPLATE.generator

    def test_smaller_fleets_are_nested_and_reproducible(self) -> None:
        generator = FleetGenerator(TEMPLATE, FleetConfig(sites=10, fleet_sizes=(10,)))
        assert generator.generate(4) == generator.generate()[:4]
        assert generator.generate() == generator.generate()
        other = FleetGenerator(TEMPLATE, FleetConfig(sites=10, fleet_sizes=(10,), seed=1))
        assert other.generate() != generator.generate()

    def test_degenerate_range_is_constant(self) -> None:
        fleet = FleetConfig(sites=3, fleet_sizes=(3,), load_scale=(1.0, 1.0))
        assert {s.load_scale for s in FleetGenerator(TEMPLATE, fleet).generate()} == {1.0}

    def test_invalid_count(self) -> None:
        with pytest.raises(ConfigurationError):
            FleetGenerator(TEMPLATE, FleetConfig()).generate(0)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
class TestSummaryStatistics:
    def test_basic_statistics(self) -> None:
        stats = SummaryStatistics.from_values([10.0, 12.0, 14.0, 16.0])
        assert stats.count == 4
        assert stats.mean == 13.0
        assert stats.minimum == 10.0
        assert stats.maximum == 16.0
        assert stats.std == pytest.approx(2.5819889)
        assert stats.coefficient_of_variation == pytest.approx(stats.std / 13.0)

    def test_bootstrap_interval_contains_mean_and_shrinks_with_n(self) -> None:
        small = SummaryStatistics.from_values([1.0, 5.0, 3.0, 4.0, 2.0])
        large = SummaryStatistics.from_values([1.0, 5.0, 3.0, 4.0, 2.0] * 20)
        assert small.contains(small.mean)
        assert large.contains(large.mean)
        assert small.ci_half_width is not None
        assert large.ci_half_width is not None
        assert large.ci_half_width < small.ci_half_width

    def test_single_value_has_no_interval(self) -> None:
        stats = SummaryStatistics.from_values([7.0])
        assert not stats.has_interval
        assert stats.ci_half_width is None
        assert not stats.contains(7.0)
        assert stats.std == 0

    def test_is_deterministic(self) -> None:
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
        assert SummaryStatistics.from_values(values) == SummaryStatistics.from_values(values)

    def test_zero_mean_has_no_cv(self) -> None:
        assert SummaryStatistics.from_values([-1.0, 1.0]).coefficient_of_variation is None

    @pytest.mark.parametrize("values", [[], [1.0, float("nan")]])
    def test_invalid_sample(self, values: list[float]) -> None:
        with pytest.raises(ConfigurationError):
            SummaryStatistics.from_values(values)


# --------------------------------------------------------------------------- #
# Fleet results
# --------------------------------------------------------------------------- #
class TestFleetResult:
    def test_savings_win_rate_and_best(self) -> None:
        fleet = FleetResult([site("a", 100, 40), site("b", 100, 60), site("c", 100, 120)])
        assert list(fleet.savings_pct("optimizer")) == [60.0, 40.0, -20.0]
        assert list(fleet.fuel_savings_l("optimizer")) == [15.0, 15.0, 15.0]
        assert fleet.win_rate("optimizer") == pytest.approx(2 / 3)
        assert fleet.best_strategy() == "optimizer"
        assert fleet.strategy_names == ["baseline", "optimizer"]
        assert fleet.candidates == ["optimizer"]
        assert len(fleet.subset(2)) == 2
        assert fleet.to_frame().shape == (6, 13)

    def test_less_reliable_strategy_is_not_recommended(self) -> None:
        fleet = FleetResult([site("a", 100, 10, cand_unmet=50)])
        assert not fleet.outcomes[0].beats("optimizer", "baseline")
        assert fleet.best_strategy() == "baseline"

    def test_validation(self) -> None:
        with pytest.raises(ConfigurationError):
            FleetResult([])
        with pytest.raises(ConfigurationError):
            FleetResult([site("a", 1, 1)], baseline="nope")
        mismatched = SiteOutcome("b", 1, 1, 0, 1, {"baseline": metrics("baseline", 1)})
        with pytest.raises(ConfigurationError, match="same strategies"):
            FleetResult([site("a", 1, 1), mismatched])
        with pytest.raises(ConfigurationError):
            FleetResult([site("a", 1, 1)]).savings_pct("magic")

    def test_zero_cost_baseline(self) -> None:
        assert site("a", 0, 0).cost_savings_pct("optimizer", "baseline") == 0.0

    def test_simulator_reports_progress(self) -> None:
        sites = FleetGenerator(TEMPLATE, FleetConfig(sites=3, fleet_sizes=(3,))).generate()
        calls: list[tuple[int, int]] = []
        fleet = FleetSimulator().run(sites, progress=lambda d, t: calls.append((d, t)))
        assert calls == [(1, 3), (2, 3), (3, 3)]
        assert [o.site_id for o in fleet.outcomes] == ["site-001", "site-002", "site-003"]
        with pytest.raises(ConfigurationError):
            FleetSimulator().run([])


# --------------------------------------------------------------------------- #
# Consistency study
# --------------------------------------------------------------------------- #
class TestConsistencyStudy:
    def test_end_to_end(self, fleet_study: tuple[FleetResult, ConsistencyReport]) -> None:
        fleet, report = fleet_study
        assert len(fleet) == 8
        assert [s.size for s in report.summaries] == [1, 4, 8]
        assert report.reference.size == 8
        assert report.reference.recommended_strategy == "optimizer"
        assert report.reference.win_rate > 0.5
        assert report.direction_consistent
        frame = report.to_frame()
        assert list(frame.index) == [1, 4, 8]
        assert bool(frame.loc[8, "agrees_with_reference"])
        assert "optimizer saves" in report.verdict()

    def test_magnitude_inconsistency_is_reported(self) -> None:
        outcomes = [site("a", 100, 10)] + [site(f"s{i}", 100, 50 + i % 3) for i in range(9)]
        report = ConsistencyStudy([1, 10]).analyse(FleetResult(outcomes))
        assert report.direction_consistent
        assert not report.magnitude_consistent
        assert report.smallest_reliable_size == 10
        assert "Magnitude differs for a fleet of 1 site:" in report.verdict()

    def test_direction_inconsistency_is_reported(self) -> None:
        outcomes = [site("a", 100, 150)] + [site(f"s{i}", 100, 50) for i in range(9)]
        report = ConsistencyStudy([1, 10]).analyse(FleetResult(outcomes))
        assert report.summaries[0].recommended_strategy == "baseline"
        assert not report.direction_consistent
        assert "NOT consistent" in report.verdict()

    def test_consistent_case(self) -> None:
        outcomes = [site(f"s{i}", 100, 50 + (i % 2)) for i in range(10)]
        report = ConsistencyStudy([2, 10]).analyse(FleetResult(outcomes))
        assert report.direction_consistent
        assert report.magnitude_consistent
        assert "Magnitude is consistent" in report.verdict()

    def test_validation(self) -> None:
        with pytest.raises(ConfigurationError):
            ConsistencyStudy([])
        with pytest.raises(ConfigurationError):
            ConsistencyStudy([5, 2])
        with pytest.raises(ConfigurationError, match="exceeds"):
            ConsistencyStudy([1, 50]).analyse(FleetResult([site("a", 1, 1)]))
        with pytest.raises(ConfigurationError):
            ConsistencyReport("optimizer", "baseline", ())

    @pytest.mark.parametrize(
        ("sizes", "text"),
        [
            ([1], "a fleet of 1 site"),
            ([5], "a fleet of 5 sites"),
            ([1, 5], "fleets of 1 and 5 sites"),
            ([1, 5, 10], "fleets of 1, 5 and 10 sites"),
        ],
    )
    def test_fleet_sizes_are_described_in_plain_english(self, sizes: list[int], text: str) -> None:
        assert describe_fleets(sizes) == text

    def test_describe_fleets_rejects_empty(self) -> None:
        with pytest.raises(ConfigurationError):
            describe_fleets([])

    def test_single_site_reference_verdict(self) -> None:
        report = ConsistencyStudy([1]).analyse(FleetResult([site("a", 100, 50)]))
        summary: FleetSizeSummary = report.reference
        assert summary.savings_pct.ci_low is None
        assert "CI" not in report.verdict()
