"""Interactive Streamlit dashboard.

Run with ``streamlit run streamlit_app.py`` (or ``hybrid-power dashboard``).
The dashboard is a thin presentation layer: it collects inputs, calls
:class:`~hybrid_power.service.SimulationService` and renders the outcome.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd
import streamlit as st

from hybrid_power.config import (
    BatteryConfig,
    FleetConfig,
    GeneratorConfig,
    GeneratorMode,
    GridConfig,
    OptimizerConfig,
    ProfileConfig,
    ScenarioConfig,
    SolarConfig,
)
from hybrid_power.data.repository import CsvProfileRepository
from hybrid_power.exceptions import HybridPowerError
from hybrid_power.fleet import FleetStudyOutcome, run_fleet_study
from hybrid_power.presentation.charts import HEATMAP_METRICS, ChartFactory
from hybrid_power.service import SimulationOutcome, SimulationService

_DECISION_COLUMNS = [
    "demand_kwh",
    "grid_available",
    "tariff",
    "solar_available_kwh",
    "battery_to_load",
    "generator_on",
    "soc",
    "unmet_kwh",
    "total_cost",
    "reasons",
]


@st.cache_data(show_spinner="Simulating strategies...", max_entries=16)
def _simulate(config_json: str, profile_csv: bytes | None) -> SimulationOutcome:
    """Run every registered strategy (cached on the serialised inputs)."""
    config = ScenarioConfig.from_json(config_json)
    provider = CsvProfileRepository(io.BytesIO(profile_csv)) if profile_csv is not None else None
    return SimulationService().run(config, provider=provider)


@st.cache_data(show_spinner="Simulating the fleet (about 0.1 s per site)...", max_entries=8)
def _simulate_fleet(template_json: str, fleet_json: str) -> FleetStudyOutcome:
    """Run a fleet consistency study (cached on the serialised inputs)."""
    template = ScenarioConfig.from_json(template_json)
    return run_fleet_study(template, FleetConfig.from_json(fleet_json))


@dataclass(frozen=True)
class KpiDelta:
    """A KPI's change against the baseline, formatted for ``st.metric``.

    Changes that round to zero at the displayed precision are shown as a
    neutral "no change" without an arrow, instead of a coloured "+0.0".
    """

    text: str
    color: Literal["normal", "inverse", "off"]
    arrow: Literal["auto", "off"]

    @classmethod
    def of(
        cls, change: float, unit: str, decimals: int = 0, *, lower_is_better: bool = True
    ) -> KpiDelta:
        """Format ``change`` (candidate minus baseline) in ``unit``."""
        if round(change, decimals) == 0:
            return cls("no change", "off", "off")
        colour: Literal["normal", "inverse"] = "inverse" if lower_is_better else "normal"
        return cls(f"{change:+,.{decimals}f} {unit}", colour, "auto")


@dataclass(frozen=True)
class DashboardInputs:
    """Validated user inputs collected from the sidebar."""

    config: ScenarioConfig
    profile_csv: bytes | None


class SidebarForm:
    """Render the sidebar controls and turn them into a :class:`ScenarioConfig`."""

    def __init__(self) -> None:
        self._defaults = ScenarioConfig()
        self._key = "default"

    def render(self) -> DashboardInputs:
        """Draw every control and return the validated inputs."""
        sidebar = st.sidebar
        sidebar.header("Scenario")
        self._load_saved_scenario()
        profile_csv = self._profile_source()
        d = self._defaults
        profile = self._profile_section(d.profile, synthetic=profile_csv is None)
        solar = self._solar_section(d.solar)
        battery = self._battery_section(d.battery)
        generator = self._generator_section(d.generator)
        grid = GridConfig(
            max_import_kw=d.grid.max_import_kw,
            emission_factor_kg_per_kwh=d.grid.emission_factor_kg_per_kwh,
        )
        optimizer = self._optimizer_section(d.optimizer)
        config = ScenarioConfig(
            name=d.name,
            currency=d.currency,
            profile=profile,
            solar=solar,
            grid=grid,
            battery=battery,
            generator=generator,
            optimizer=optimizer,
        )
        return DashboardInputs(config=config, profile_csv=profile_csv)

    # ------------------------------------------------------------------ #
    # Sections
    # ------------------------------------------------------------------ #
    def _k(self, name: str) -> str:
        """Widget key scoped to the loaded scenario so uploads reset the widgets."""
        return f"{self._key}:{name}"

    def _load_saved_scenario(self) -> None:
        upload = st.sidebar.file_uploader("Load scenario (JSON)", type=["json"])
        if upload is None:
            return
        raw = upload.getvalue()
        self._defaults = ScenarioConfig.from_json(raw.decode("utf-8"))
        self._key = hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:8]

    def _profile_source(self) -> bytes | None:
        choice = st.sidebar.radio(
            "Operating profile",
            ["Synthetic (configurable)", "Upload CSV"],
            help="CSV columns: timestamp, demand_kw, solar_factor, grid_available, tariff_per_kwh",
        )
        if choice == "Upload CSV":
            upload = st.sidebar.file_uploader("Profile CSV", type=["csv"])
            if upload is None:
                st.sidebar.info("Upload a profile CSV, or switch back to the synthetic profile.")
                st.stop()
            return upload.getvalue()
        return None

    def _profile_section(self, d: ProfileConfig, *, synthetic: bool) -> ProfileConfig:
        if not synthetic:
            return d
        with st.sidebar.expander("Demand, solar resource and grid", expanded=True):
            days = st.slider("Simulation days", 1, 90, d.days, key=self._k("days"))
            base, peak = st.slider(
                "Site load range (kW)",
                0.5,
                12.0,
                (d.base_load_kw, d.peak_load_kw),
                step=0.1,
                key=self._k("load"),
            )
            clearness = st.slider(
                "Mean sky clearness", 0.1, 1.0, d.solar_mean_clearness, 0.05, key=self._k("clear")
            )
            variability = st.slider(
                "Day-to-day cloud variability",
                0.0,
                0.9,
                d.solar_variability,
                0.05,
                key=self._k("var"),
            )
            outages = st.slider(
                "Grid outages per day", 0.0, 6.0, d.outages_per_day, 0.1, key=self._k("outages")
            )
            duration = st.slider(
                "Mean outage duration (h)", 1.0, 12.0, d.mean_outage_hours, 0.5, key=self._k("dur")
            )
            offpeak = st.number_input(
                "Off-peak tariff (per kWh)", 0.0, 2.0, d.offpeak_tariff, 0.01, key=self._k("off")
            )
            peak_price = st.number_input(
                "Peak tariff (per kWh)", 0.0, 2.0, d.peak_tariff, 0.01, key=self._k("peak")
            )
            peak_start, peak_end = st.slider(
                "Peak tariff hours",
                0,
                24,
                (d.peak_start_hour, d.peak_end_hour),
                key=self._k("peakh"),
            )
            seed = st.number_input("Random seed", 0, 1_000_000, d.seed, key=self._k("seed"))
        return ProfileConfig(
            days=days,
            start_date=d.start_date,
            base_load_kw=base,
            peak_load_kw=peak,
            load_noise=d.load_noise,
            solar_mean_clearness=clearness,
            solar_variability=variability,
            outages_per_day=outages,
            mean_outage_hours=duration,
            peak_tariff=float(peak_price),
            offpeak_tariff=float(offpeak),
            peak_start_hour=min(peak_start, 23),
            peak_end_hour=peak_end,
            seed=int(seed),
        )

    def _solar_section(self, d: SolarConfig) -> SolarConfig:
        with st.sidebar.expander("Solar PV"):
            rated = st.slider("PV array size (kWp)", 0.0, 30.0, d.rated_kw, 0.5, key=self._k("pv"))
        return SolarConfig(rated_kw=rated, om_cost_per_kwh=d.om_cost_per_kwh)

    def _battery_section(self, d: BatteryConfig) -> BatteryConfig:
        with st.sidebar.expander("Battery"):
            capacity = st.slider(
                "Capacity (kWh)", 5.0, 120.0, d.capacity_kwh, 1.0, key=self._k("cap")
            )
            min_soc, max_soc = st.slider(
                "Allowed SoC window", 0.0, 1.0, (d.min_soc, d.max_soc), 0.05, key=self._k("win")
            )
            initial = st.slider(
                "Initial state of charge", 0.0, 1.0, d.initial_soc, 0.05, key=self._k("init")
            )
            power = st.slider(
                "Max charge/discharge (kW)", 1.0, 40.0, d.max_charge_kw, 0.5, key=self._k("bpow")
            )
        return BatteryConfig(
            capacity_kwh=capacity,
            initial_soc=min(initial, max_soc),
            min_soc=min_soc,
            max_soc=max_soc,
            charge_efficiency=d.charge_efficiency,
            discharge_efficiency=d.discharge_efficiency,
            max_charge_kw=power,
            max_discharge_kw=power,
            degradation_cost_per_kwh=d.degradation_cost_per_kwh,
        )

    def _generator_section(self, d: GeneratorConfig) -> GeneratorConfig:
        with st.sidebar.expander("Diesel generator"):
            rated = st.slider("Rated power (kW)", 2.0, 40.0, d.rated_kw, 0.5, key=self._k("gen"))
            tank = st.slider(
                "Tank capacity (L)", 20.0, 1000.0, d.tank_capacity_l, 10.0, key=self._k("tank")
            )
            fuel = st.slider(
                "Initial fuel (L)", 0.0, tank, min(d.initial_fuel_l, tank), 5.0, key=self._k("fuel")
            )
            price = st.number_input(
                "Diesel price (per L)", 0.0, 5.0, d.fuel_price_per_l, 0.05, key=self._k("price")
            )
            default_days = (d.refuel_interval_hours or 0) // 24
            refuel_days = st.number_input(
                "Refuel every N days (0 = never)", 0, 60, default_days, key=self._k("refuel")
            )
        return GeneratorConfig(
            rated_kw=rated,
            tank_capacity_l=tank,
            initial_fuel_l=fuel,
            fuel_price_per_l=float(price),
            min_load_ratio=d.min_load_ratio,
            fuel_curve_slope=d.fuel_curve_slope,
            fuel_curve_intercept=d.fuel_curve_intercept,
            maintenance_cost_per_hour=d.maintenance_cost_per_hour,
            co2_kg_per_litre=d.co2_kg_per_litre,
            refuel_interval_hours=int(refuel_days) * 24 if refuel_days else None,
        )

    def _optimizer_section(self, d: OptimizerConfig) -> OptimizerConfig:
        with st.sidebar.expander("Optimiser rules"):
            modes = list(GeneratorMode)
            mode = st.selectbox(
                "Generator control",
                modes,
                index=modes.index(d.generator_mode),
                format_func=lambda m: m.value.replace("_", " ").title(),
                key=self._k("mode"),
            )
            reserve = st.slider(
                "Outage reserve SoC", 0.0, 1.0, d.reserve_soc, 0.05, key=self._k("reserve")
            )
            target = st.slider(
                "Off-peak grid-charge target SoC",
                0.0,
                1.0,
                d.grid_charge_target_soc,
                0.05,
                key=self._k("target"),
            )
            peak_threshold = st.number_input(
                "Use battery when tariff >=",
                0.0,
                2.0,
                d.peak_tariff_threshold,
                0.01,
                key=self._k("pt"),
            )
            charge_threshold = st.number_input(
                "Charge from grid when tariff <=",
                0.0,
                2.0,
                d.grid_charge_tariff_threshold,
                0.01,
                key=self._k("ct"),
            )
        return OptimizerConfig(
            reserve_soc=reserve,
            peak_tariff_threshold=float(peak_threshold),
            grid_charge_tariff_threshold=float(charge_threshold),
            grid_charge_target_soc=target,
            generator_mode=mode,
            cycle_charge_load_ratio=d.cycle_charge_load_ratio,
            generator_stop_soc=d.generator_stop_soc,
        )


class ResultsView:
    """Render KPIs, charts, the decision log and exports for an outcome."""

    def __init__(self, outcome: SimulationOutcome, candidate: str = "optimizer") -> None:
        self._outcome = outcome
        self._report = outcome.report
        self._candidate = (
            candidate if candidate in outcome.results else outcome.report.best().strategy_name
        )
        self._charts = ChartFactory(currency=outcome.config.currency)

    def render(self) -> None:
        """Draw the full results page."""
        self._render_kpis()
        tabs = st.tabs(
            [
                "Comparison",
                "Power flow",
                "Heatmaps",
                "Decision log",
                "Fleet study",
                "Input profile",
                "Export",
            ]
        )
        with tabs[0]:
            self._render_comparison()
        with tabs[1]:
            self._render_power_flow()
        with tabs[2]:
            self._render_heatmaps()
        with tabs[3]:
            self._render_decision_log()
        with tabs[4]:
            FleetStudyView(self._outcome.config, self._charts).render()
        with tabs[5]:
            self._render_profile()
        with tabs[6]:
            self._render_exports()

    def _render_kpis(self) -> None:
        comparison = self._report.comparison(self._candidate)
        c, b = comparison.candidate, comparison.baseline
        currency = self._outcome.config.currency
        st.subheader(f"`{self._candidate}` vs `{self._report.baseline_name}`")
        kpis = (
            (
                "Operating cost",
                f"{c.total_cost:,.0f} {currency}",
                KpiDelta.of(-comparison.cost_savings, currency),
            ),
            (
                "Diesel used",
                f"{c.fuel_used_l:,.0f} L",
                KpiDelta.of(-comparison.fuel_savings_l, "L"),
            ),
            (
                "CO2 emitted",
                f"{c.emissions_kg:,.0f} kg",
                KpiDelta.of(-comparison.emissions_reduction_kg, "kg"),
            ),
            (
                "Renewable share",
                f"{c.renewable_fraction:.0%}",
                KpiDelta.of(
                    (c.renewable_fraction - b.renewable_fraction) * 100, "pp", lower_is_better=False
                ),
            ),
            (
                "Generator hours",
                f"{c.generator_hours:,.0f} h",
                KpiDelta.of(-comparison.generator_hours_reduction, "h"),
            ),
            (
                "Unmet demand",
                f"{c.unmet_energy_kwh:,.1f} kWh",
                KpiDelta.of(comparison.unmet_energy_change_kwh, "kWh", decimals=1),
            ),
        )
        # Two rows of three keep values readable on narrower screens.
        for row in (kpis[:3], kpis[3:]):
            for column, (label, value, delta) in zip(st.columns(len(row)), row, strict=True):
                column.metric(
                    label,
                    value,
                    delta.text,
                    delta_color=delta.color,
                    delta_arrow=delta.arrow,
                )
        best = self._report.best()
        pct = self._report.comparison(best.strategy_name).cost_savings_pct
        saving = f" - {pct:.1f}% cheaper than the baseline" if pct is not None else ""
        st.info(
            f"Recommended strategy: **{best.strategy_name}** "
            f"(most reliable, then cheapest){saving}."
        )

    def _render_comparison(self) -> None:
        metrics = list(self._report.metrics.values())
        left, right = st.columns(2)
        left.plotly_chart(self._charts.cost_breakdown(metrics))
        right.plotly_chart(self._charts.energy_mix(metrics))
        st.plotly_chart(self._charts.environmental_impact(metrics))
        st.plotly_chart(self._charts.cumulative_cost(self._outcome.results))
        st.dataframe(self._charts.metrics_table(metrics).style.format("{:,.2f}"))
        with st.expander("Strategy descriptions"):
            for result in self._outcome.results.values():
                st.markdown(f"- **{result.strategy_name}** - {result.strategy_description}")

    def _strategy_picker(self, key: str) -> str:
        names = list(self._outcome.results)
        return str(st.selectbox("Strategy", names, index=names.index(self._candidate), key=key))

    def _day_window(self, key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        stamps = self._outcome.profile.timestamps
        days = max(1, len(stamps) // 24)
        first, last = st.slider("Days", 1, days, (1, min(days, 7)), key=key) if days > 1 else (1, 1)
        start = stamps[0] + pd.Timedelta(days=first - 1)
        end = stamps[0] + pd.Timedelta(days=last)
        return start, end

    def _render_power_flow(self) -> None:
        name = self._strategy_picker("flow_strategy")
        start, end = self._day_window("flow_days")
        result = self._outcome.results[name]
        try:
            windowed = result.between(start.to_pydatetime(), end.to_pydatetime())
        except HybridPowerError:
            st.warning("No data in the selected window.")
            return
        st.plotly_chart(self._charts.power_flow(windowed))
        st.plotly_chart(self._charts.battery_soc(self._outcome.results))
        st.plotly_chart(self._charts.daily_energy(result))

    def _render_heatmaps(self) -> None:
        col1, col2 = st.columns(2)
        with col1:
            name = self._strategy_picker("heat_strategy")
        metric = str(col2.selectbox("Metric", list(HEATMAP_METRICS), index=1, key="heat_metric"))
        result = self._outcome.results[name]
        st.plotly_chart(self._charts.calendar_heatmap(result, metric))
        st.plotly_chart(self._charts.hourly_heatmap(result, metric))
        baseline = self._outcome.results[self._report.baseline_name]
        if name != self._report.baseline_name:
            st.plotly_chart(self._charts.savings_calendar(baseline, result))

    def _render_decision_log(self) -> None:
        st.caption("Every step records why each source was chosen.")
        name = self._strategy_picker("log_strategy")
        frame = self._outcome.results[name].frame
        col1, col2 = st.columns(2)
        outages_only = col1.checkbox("Only grid outage hours")
        unmet_only = col2.checkbox("Only hours with unmet demand")
        if outages_only:
            frame = frame[~frame["grid_available"]]
        if unmet_only:
            frame = frame[frame["unmet_kwh"] > 0]
        st.dataframe(frame[_DECISION_COLUMNS], height=480)

    def _render_profile(self) -> None:
        profile = self._outcome.profile
        cols = st.columns(4)
        cols[0].metric("Steps", f"{len(profile):,}")
        cols[1].metric(
            "Energy demand", f"{profile.load.total_energy_kwh(profile.step_hours):,.0f} kWh"
        )
        cols[2].metric("Grid availability", f"{profile.grid_availability_ratio:.1%}")
        cols[3].metric("Outage hours", f"{profile.outage_hours:,.0f} h")
        st.plotly_chart(
            self._charts.profile_overview(profile, self._outcome.config.solar.rated_kw),
        )

    def _render_exports(self) -> None:
        st.download_button(
            "Scenario (JSON)",
            self._outcome.config.to_json(),
            file_name="scenario.json",
            mime="application/json",
        )
        st.download_button(
            "Operating profile (CSV)",
            self._outcome.profile.to_frame().to_csv(index=False),
            file_name="profile.csv",
            mime="text/csv",
        )
        st.download_button(
            "KPI summary (CSV)",
            self._report.to_frame().to_csv(),
            file_name="metrics.csv",
            mime="text/csv",
        )
        for name, result in self._outcome.results.items():
            st.download_button(
                f"Hourly results - {name} (CSV)",
                result.frame.to_csv(),
                file_name=f"results_{name}.csv",
                mime="text/csv",
                key=f"dl_{name}",
            )


class FleetStudyView:
    """Inputs and results of the few-versus-many base stations study."""

    def __init__(self, template: ScenarioConfig, charts: ChartFactory) -> None:
        self._template = template
        self._charts = charts

    def render(self) -> None:
        """Draw the study form and, once run, its results."""
        st.markdown(
            "Every site shares the equipment in the sidebar but gets its own demand level, "
            "grid reliability and sky conditions. Smaller fleets are the first sites of the "
            "largest one, so each size can be compared directly with the others."
        )
        with st.form("fleet_form"):
            col1, col2, col3 = st.columns(3)
            sites = col1.slider("Base stations (largest fleet)", 5, 100, 50, 5)
            days = col2.slider(
                "Days simulated per site", 7, 60, min(30, self._template.profile.days)
            )
            seed = col3.number_input("Fleet seed", 0, 1_000_000, 2026)
            col4, col5 = st.columns(2)
            load = col4.slider("Load scale range", 0.3, 2.5, (0.7, 1.4), 0.05)
            clearness = col5.slider("Sky clearness range", 0.1, 1.0, (0.5, 0.85), 0.05)
            outages = col4.slider("Outages per day range", 0.0, 8.0, (0.2, 3.0), 0.1)
            duration = col5.slider("Mean outage hours range", 1.0, 24.0, (1.0, 6.0), 0.5)
            submitted = st.form_submit_button("Run fleet study", type="primary")
        if submitted:
            fleet = FleetConfig.with_sites(
                sites,
                load_scale=list(load),
                solar_mean_clearness=list(clearness),
                outages_per_day=list(outages),
                mean_outage_hours=list(duration),
                seed=int(seed),
            )
            template = replace(self._template, profile=replace(self._template.profile, days=days))
            st.session_state["fleet_request"] = (template.to_json(), fleet.to_json())
        request = st.session_state.get("fleet_request")
        if request is None:
            st.info("Choose the fleet settings and press **Run fleet study**.")
            return
        self._render_results(_simulate_fleet(*request))

    def _render_results(self, study: FleetStudyOutcome) -> None:
        report, fleet = study.report, study.fleet
        verdict = report.verdict()
        if report.direction_consistent and report.magnitude_consistent:
            st.success(verdict)
        elif report.direction_consistent:
            st.warning(verdict)
        else:
            st.error(verdict)
        ref = report.reference
        cols = st.columns(4)
        cols[0].metric("Sites simulated", f"{len(fleet)}")
        cols[1].metric("Mean cost saving", f"{ref.savings_pct.mean:.1f}%")
        half = ref.savings_pct.ci_half_width
        cols[2].metric("95% CI half-width", f"±{half:.1f} pp" if half is not None else "n/a")
        cols[3].metric("Optimiser win rate", f"{ref.win_rate:.0%}")
        st.plotly_chart(self._charts.fleet_convergence(report))
        left, right = st.columns(2)
        left.plotly_chart(self._charts.fleet_distribution(fleet))
        right.plotly_chart(self._charts.fleet_drivers(fleet, report.candidate))
        st.dataframe(
            report.to_frame().style.format(
                {
                    "mean_saving_pct": "{:.1f}",
                    "std_saving_pct": "{:.1f}",
                    "ci_low_pct": "{:.1f}",
                    "ci_high_pct": "{:.1f}",
                    "min_saving_pct": "{:.1f}",
                    "max_saving_pct": "{:.1f}",
                    "mean_fuel_saved_l": "{:.1f}",
                    "win_rate": "{:.0%}",
                },
                na_rep="-",
            )
        )
        st.download_button(
            "Per-site results (CSV)",
            fleet.to_frame().to_csv(index=False),
            file_name="fleet_sites.csv",
            mime="text/csv",
        )


class DashboardApp:
    """Top-level Streamlit application."""

    TITLE = "Hybrid Power Optimiser"

    def __init__(self, form: SidebarForm | None = None) -> None:
        self._form = form or SidebarForm()

    def run(self) -> None:
        """Render the page."""
        st.set_page_config(page_title=self.TITLE, page_icon=":zap:", layout="wide")
        st.title(self.TITLE)
        st.caption(
            "Decision-support simulator for a telecom base station with solar PV, grid, "
            "battery storage and a diesel generator. It does not control real equipment."
        )
        try:
            inputs = self._form.render()
            outcome = _simulate(inputs.config.to_json(), inputs.profile_csv)
        except HybridPowerError as exc:
            st.error(f"Invalid input: {exc}")
            st.stop()
        ResultsView(outcome).render()


def main() -> None:
    """Entry point used by ``streamlit_app.py``."""
    DashboardApp().run()
