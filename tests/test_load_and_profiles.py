from __future__ import annotations

import io
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hybrid_power.config import ProfileConfig
from hybrid_power.data.profiles import ProfileSet
from hybrid_power.data.repository import CsvProfileRepository
from hybrid_power.data.synthetic import SyntheticProfileGenerator
from hybrid_power.domain.conditions import OperatingConditions
from hybrid_power.domain.load import SiteLoad
from hybrid_power.exceptions import ProfileDataError


class TestSiteLoad:
    def test_statistics(self) -> None:
        load = SiteLoad([1.0, 2.0, 3.0])
        assert (load.peak_kw, load.average_kw, load.minimum_kw) == (3.0, 2.0, 1.0)
        assert load.total_energy_kwh(0.5) == 3.0
        assert list(load) == [1.0, 2.0, 3.0]
        assert len(load) == 3
        assert load.demand_at(1) == 2.0

    def test_values_are_read_only(self) -> None:
        load = SiteLoad([1.0, 2.0])
        with pytest.raises(ValueError, match="read-only"):
            load.values[0] = 5.0

    def test_scaled_returns_new_instance(self) -> None:
        load = SiteLoad([1.0, 2.0])
        doubled = load.scaled(2)
        assert list(doubled) == [2.0, 4.0]
        assert list(load) == [1.0, 2.0]
        assert doubled != load
        assert SiteLoad([1.0, 2.0]) == load
        assert hash(SiteLoad([1.0, 2.0])) == hash(load)

    @pytest.mark.parametrize("values", [[], [1.0, -1.0], [1.0, float("nan")], [[1.0]], ["x"]])
    def test_invalid_values(self, values: list[object]) -> None:
        with pytest.raises(ProfileDataError):
            SiteLoad(values)  # type: ignore[arg-type]

    def test_out_of_range_index(self) -> None:
        with pytest.raises(ProfileDataError):
            SiteLoad([1.0]).demand_at(3)


class TestOperatingConditions:
    def test_derived_values(self) -> None:
        conditions = OperatingConditions(0, datetime(2026, 1, 1, 14), 4.0, 0.5, True, 0.2, 0.5)
        assert conditions.demand_kwh == 2.0
        assert conditions.hour_of_day == 14

    @pytest.mark.parametrize(
        "kwargs",
        [{"index": -1}, {"demand_kw": -1.0}, {"solar_factor": 1.5}, {"step_hours": 0.0}],
    )
    def test_validation(self, kwargs: dict[str, float]) -> None:
        params: dict[str, object] = {
            "index": 0,
            "timestamp": datetime(2026, 1, 1),
            "demand_kw": 1.0,
            "solar_factor": 0.5,
            "grid_available": True,
            "tariff_per_kwh": 0.1,
            "step_hours": 1.0,
        }
        params.update(kwargs)
        with pytest.raises(ProfileDataError):
            OperatingConditions(**params)  # type: ignore[arg-type]


class TestSyntheticProfileGenerator:
    def test_shape_and_ranges(self) -> None:
        config = ProfileConfig(days=5, seed=3)
        profile = SyntheticProfileGenerator(config).get_profile()
        assert len(profile) == 120
        assert profile.step_hours == 1.0
        assert np.all((profile.solar_factor >= 0) & (profile.solar_factor <= 1))
        assert set(np.unique(profile.tariff_per_kwh)) <= {config.peak_tariff, config.offpeak_tariff}
        assert profile.load.minimum_kw >= 0

    def test_no_solar_at_night(self) -> None:
        profile = SyntheticProfileGenerator(ProfileConfig(days=3)).get_profile()
        hours = profile.timestamps.hour
        night = (hours < 6) | (hours >= 18)
        assert np.all(profile.solar_factor[night] == 0)
        assert np.max(profile.solar_factor[~night]) > 0

    def test_is_reproducible_for_same_seed(self) -> None:
        config = ProfileConfig(days=4, seed=11)
        first = SyntheticProfileGenerator(config).get_profile().to_frame()
        second = SyntheticProfileGenerator(config).get_profile().to_frame()
        pd.testing.assert_frame_equal(first, second)
        third = SyntheticProfileGenerator(replace(config, seed=12)).get_profile().to_frame()
        assert not first.equals(third)

    def test_outage_parameters_control_availability(self) -> None:
        reliable = SyntheticProfileGenerator(
            ProfileConfig(days=30, outages_per_day=0)
        ).get_profile()
        assert reliable.grid_availability_ratio == 1.0
        assert reliable.outage_hours == 0
        weak = SyntheticProfileGenerator(
            ProfileConfig(days=30, outages_per_day=4, mean_outage_hours=6)
        ).get_profile()
        assert weak.grid_availability_ratio < 0.7

    def test_constant_clearness_when_no_variability(self) -> None:
        config = ProfileConfig(days=2, solar_variability=0.0, solar_mean_clearness=0.5)
        profile = SyntheticProfileGenerator(config).get_profile()
        noon = profile.timestamps.hour == 12
        assert np.allclose(profile.solar_factor[noon], 0.5, atol=0.2)

    def test_peak_window_wrapping_midnight(self) -> None:
        config = ProfileConfig(days=1, peak_start_hour=22, peak_end_hour=2)
        profile = SyntheticProfileGenerator(config).get_profile()
        hours = profile.timestamps.hour
        peak = profile.tariff_per_kwh == config.peak_tariff
        assert set(hours[peak]) == {22, 23, 0, 1}


class TestProfileSetAndRepository:
    def test_iteration_yields_conditions(self) -> None:
        profile = SyntheticProfileGenerator(ProfileConfig(days=1)).get_profile()
        steps = list(profile)
        assert len(steps) == 24
        assert steps[5].index == 5
        assert steps[5].demand_kw == pytest.approx(profile.load.values[5])
        with pytest.raises(ProfileDataError):
            profile.conditions_at(24)

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        profile = SyntheticProfileGenerator(ProfileConfig(days=2)).get_profile()
        path = CsvProfileRepository.save(profile, tmp_path / "nested" / "profile.csv")
        loaded = CsvProfileRepository(path).get_profile()
        assert len(loaded) == len(profile)
        assert np.allclose(loaded.load.values, profile.load.values, atol=1e-4)
        assert np.array_equal(loaded.grid_available, profile.grid_available)
        assert "ProfileSet" in repr(loaded)

    def test_reads_in_memory_buffer_with_text_booleans(self) -> None:
        csv = (
            "timestamp,demand_kw,solar_factor,grid_available,tariff_per_kwh\n"
            "2026-01-01 00:00,3,0,yes,0.1\n"
            "2026-01-01 00:30,3,0,no,0.1\n"
        )
        profile = CsvProfileRepository(io.StringIO(csv)).get_profile()
        assert profile.step_hours == 0.5
        assert list(profile.grid_available) == [True, False]

    @pytest.mark.parametrize(
        ("csv", "message"),
        [
            ("timestamp,demand_kw\n2026-01-01,3\n", "missing column"),
            (
                "timestamp,demand_kw,solar_factor,grid_available,tariff_per_kwh\n"
                "2026-01-01 00:00,3,0,maybe,0.1\n",
                "not a boolean",
            ),
            (
                "timestamp,demand_kw,solar_factor,grid_available,tariff_per_kwh\n"
                "2026-01-01 00:00,3,1.5,1,0.1\n",
                "between 0 and 1",
            ),
            (
                "timestamp,demand_kw,solar_factor,grid_available,tariff_per_kwh\n"
                "2026-01-01 00:00,3,0,1,0.1\n2026-01-01 01:00,3,0,1,0.1\n"
                "2026-01-01 03:00,3,0,1,0.1\n",
                "evenly spaced",
            ),
            ("", "Could not parse"),
        ],
    )
    def test_malformed_csv_is_rejected(self, csv: str, message: str) -> None:
        with pytest.raises(ProfileDataError, match=message):
            CsvProfileRepository(io.StringIO(csv)).get_profile()

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileDataError, match="not found"):
            CsvProfileRepository(tmp_path / "nope.csv").get_profile()

    def test_length_mismatch_is_rejected(self) -> None:
        stamps = pd.date_range("2026-01-01", periods=3, freq="h")
        with pytest.raises(ProfileDataError, match="expected 3"):
            ProfileSet(stamps, SiteLoad([1, 2, 3]), [0, 0], [True] * 3, [0.1] * 3)
        with pytest.raises(ProfileDataError, match="DatetimeIndex"):
            ProfileSet(stamps[:2], SiteLoad([1, 2, 3]), [0] * 3, [True] * 3, [0.1] * 3)
        with pytest.raises(ProfileDataError, match="non-negative"):
            ProfileSet(stamps, SiteLoad([1, 2, 3]), [0] * 3, [True] * 3, [-0.1] * 3)
