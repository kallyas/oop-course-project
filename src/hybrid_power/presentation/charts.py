"""Plotly figure factory for the dashboard and reports.

Keeping chart construction in a plain class (rather than inside Streamlit
callbacks) makes the figures reusable outside the dashboard and testable
without a running UI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hybrid_power.analytics.metrics import PerformanceMetrics
from hybrid_power.data.profiles import ProfileSet
from hybrid_power.exceptions import ConfigurationError
from hybrid_power.fleet.consistency import ConsistencyReport
from hybrid_power.fleet.simulator import FleetResult
from hybrid_power.simulation.engine import SimulationResult

#: Consistent colour per energy path across every chart.
SOURCE_COLOURS: Mapping[str, str] = {
    "solar": "#E9A23B",
    "battery": "#3A7CA5",
    "grid": "#7A8691",
    "generator": "#C8553D",
    "unmet": "#2B2D42",
    "demand": "#111111",
}

#: Colour per built-in strategy; unknown strategies fall back to Plotly's palette.
STRATEGY_COLOURS: Mapping[str, str] = {
    "baseline": "#7A8691",
    "solar_first": "#E9A23B",
    "optimizer": "#2A9D8F",
}

_FLOW_SERIES: tuple[tuple[str, str, str], ...] = (
    ("solar_to_load", "Solar", "solar"),
    ("battery_to_load", "Battery", "battery"),
    ("grid_to_load", "Grid", "grid"),
    ("generator_to_load", "Generator", "generator"),
    ("unmet_kwh", "Unmet", "unmet"),
)

#: Metrics available to the heatmaps: label -> (column, colour scale).
HEATMAP_METRICS: Mapping[str, tuple[str, str]] = {
    "Energy demand (kWh)": ("demand_kwh", "Blues"),
    "Operating cost": ("total_cost", "YlOrRd"),
    "Grid import (kWh)": ("grid_import", "Greys"),
    "Generator output (kWh)": ("generator_output", "Reds"),
    "Solar used (kWh)": ("solar_used", "YlOrBr"),
    "Unmet demand (kWh)": ("unmet_kwh", "Purples"),
}

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _with_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the aggregate energy columns used by the heatmaps."""
    derived: pd.DataFrame = frame.copy()
    derived["grid_import"] = derived["grid_to_load"] + derived["grid_to_battery"]
    derived["generator_output"] = (
        derived["generator_to_load"] + derived["generator_to_battery"] + derived["generator_dumped"]
    )
    derived["solar_used"] = derived["solar_to_load"] + derived["solar_to_battery"]
    return derived


def _metric(label: str) -> tuple[str, str]:
    try:
        return HEATMAP_METRICS[label]
    except KeyError as exc:
        options = ", ".join(HEATMAP_METRICS)
        raise ConfigurationError(
            f"Unknown heatmap metric {label!r}; choose from: {options}"
        ) from exc


class ChartFactory:
    """Build Plotly figures from profiles, results and metrics."""

    def __init__(self, currency: str = "USD", template: str = "plotly_white") -> None:
        self._currency = currency
        self._template = template

    def _finish(self, figure: go.Figure, title: str, height: int = 420) -> go.Figure:
        figure.update_layout(
            title=title,
            template=self._template,
            height=height,
            hovermode="x unified",
            # Below the plot, so a legend that wraps never covers the title.
            legend={"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0},
            margin={"l": 40, "r": 20, "t": 60, "b": 100},
        )
        return figure

    @staticmethod
    def _strategy_colour(name: str) -> str | None:
        return STRATEGY_COLOURS.get(name)

    # ------------------------------------------------------------------ #
    # Input data
    # ------------------------------------------------------------------ #
    def profile_overview(self, profile: ProfileSet, solar_rated_kw: float) -> go.Figure:
        """Demand vs. PV potential, grid availability and tariff on a shared time axis."""
        frame = profile.to_frame()
        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.55, 0.2, 0.25],
            vertical_spacing=0.05,
            subplot_titles=("Demand and PV potential (kW)", "Grid available", "Tariff"),
        )
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame["demand_kw"],
                name="Demand",
                line={"color": SOURCE_COLOURS["demand"], "width": 1.2},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame["solar_factor"] * solar_rated_kw,
                name="PV potential",
                fill="tozeroy",
                line={"color": SOURCE_COLOURS["solar"], "width": 0.8},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame["grid_available"].astype(int),
                name="Grid on",
                line={"color": SOURCE_COLOURS["grid"], "shape": "hv"},
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame["tariff_per_kwh"],
                name=f"Tariff ({self._currency}/kWh)",
                line={"color": "#6A4C93", "shape": "hv"},
            ),
            row=3,
            col=1,
        )
        figure.update_yaxes(tickvals=[0, 1], ticktext=["off", "on"], row=2, col=1)
        return self._finish(figure, "Operating profile", height=560)

    # ------------------------------------------------------------------ #
    # Single-strategy views
    # ------------------------------------------------------------------ #
    def power_flow(self, result: SimulationResult) -> go.Figure:
        """Hourly stacked area of how the load was served."""
        frame = result.frame
        figure = go.Figure()
        for column, label, colour_key in _FLOW_SERIES:
            figure.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame[column],
                    name=label,
                    stackgroup="load",
                    line={"width": 0, "color": SOURCE_COLOURS[colour_key]},
                    hovertemplate="%{y:.2f} kWh",
                )
            )
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["demand_kwh"],
                name="Demand",
                line={"color": SOURCE_COLOURS["demand"], "width": 1, "dash": "dot"},
                hovertemplate="%{y:.2f} kWh",
            )
        )
        figure.update_yaxes(title_text="Energy per step (kWh)")
        return self._finish(figure, f"Power flow - {result.strategy_name}")

    def daily_energy(self, result: SimulationResult) -> go.Figure:
        """Daily energy served by each source."""
        columns = [column for column, _, _ in _FLOW_SERIES]
        daily = result.frame[columns].resample("D").sum()
        figure = go.Figure()
        for column, label, colour_key in _FLOW_SERIES:
            figure.add_trace(
                go.Bar(
                    x=daily.index,
                    y=daily[column],
                    name=label,
                    marker_color=SOURCE_COLOURS[colour_key],
                )
            )
        figure.update_layout(barmode="stack")
        figure.update_yaxes(title_text="kWh per day")
        return self._finish(figure, f"Daily energy by source - {result.strategy_name}")

    # ------------------------------------------------------------------ #
    # Multi-strategy comparisons
    # ------------------------------------------------------------------ #
    def battery_soc(self, results: Mapping[str, SimulationResult]) -> go.Figure:
        """Battery state of charge for each strategy."""
        figure = go.Figure()
        for name, result in results.items():
            figure.add_trace(
                go.Scatter(
                    x=result.frame.index,
                    y=result.frame["soc"] * 100,
                    name=name,
                    line={"color": self._strategy_colour(name), "width": 1.4},
                    hovertemplate="%{y:.1f}%",
                )
            )
        figure.update_yaxes(title_text="State of charge (%)", range=[0, 100])
        return self._finish(figure, "Battery state of charge")

    def cumulative_cost(self, results: Mapping[str, SimulationResult]) -> go.Figure:
        """Running total of operating cost for each strategy."""
        figure = go.Figure()
        for name, result in results.items():
            figure.add_trace(
                go.Scatter(
                    x=result.frame.index,
                    y=result.frame["total_cost"].cumsum(),
                    name=name,
                    line={"color": self._strategy_colour(name), "width": 2},
                    hovertemplate=f"%{{y:,.2f}} {self._currency}",
                )
            )
        figure.update_yaxes(title_text=f"Cumulative cost ({self._currency})")
        return self._finish(figure, "Cumulative operating cost")

    def cost_breakdown(self, metrics: Sequence[PerformanceMetrics]) -> go.Figure:
        """Stacked operating cost components per strategy."""
        names = [m.strategy_name for m in metrics]
        parts = (
            ("Grid energy", [m.grid_cost for m in metrics], SOURCE_COLOURS["grid"]),
            (
                "Diesel + maintenance",
                [m.generator_cost for m in metrics],
                SOURCE_COLOURS["generator"],
            ),
            ("Battery wear", [m.battery_cost for m in metrics], SOURCE_COLOURS["battery"]),
        )
        figure = go.Figure(
            [
                go.Bar(x=names, y=values, name=label, marker_color=colour)
                for label, values, colour in parts
            ]
        )
        figure.update_layout(barmode="stack")
        figure.update_yaxes(title_text=self._currency)
        return self._finish(figure, "Operating cost breakdown")

    def energy_mix(self, metrics: Sequence[PerformanceMetrics]) -> go.Figure:
        """Energy generated by each source per strategy."""
        names = [m.strategy_name for m in metrics]
        parts = (
            ("Solar", [m.solar_used_kwh for m in metrics], SOURCE_COLOURS["solar"]),
            ("Grid", [m.grid_energy_kwh for m in metrics], SOURCE_COLOURS["grid"]),
            ("Generator", [m.generator_energy_kwh for m in metrics], SOURCE_COLOURS["generator"]),
        )
        figure = go.Figure(
            [
                go.Bar(x=names, y=values, name=label, marker_color=colour)
                for label, values, colour in parts
            ]
        )
        figure.update_layout(barmode="stack")
        figure.update_yaxes(title_text="kWh")
        return self._finish(figure, "Energy mix")

    def environmental_impact(self, metrics: Sequence[PerformanceMetrics]) -> go.Figure:
        """Diesel use, generator hours and CO2 per strategy, side by side."""
        names = [m.strategy_name for m in metrics]
        colours = [self._strategy_colour(name) for name in names]
        figure = make_subplots(
            rows=1, cols=3, subplot_titles=("Diesel (L)", "Generator hours", "CO2 (kg)")
        )
        series = (
            [m.fuel_used_l for m in metrics],
            [m.generator_hours for m in metrics],
            [m.emissions_kg for m in metrics],
        )
        for column, values in enumerate(series, start=1):
            figure.add_trace(
                go.Bar(x=names, y=values, marker_color=colours, showlegend=False),
                row=1,
                col=column,
            )
        return self._finish(figure, "Fuel, run-hours and emissions", height=380)

    @staticmethod
    def metrics_table(metrics: Sequence[PerformanceMetrics]) -> pd.DataFrame:
        """Human-friendly KPI table (strategies as columns)."""
        rows = {
            m.strategy_name: {
                "Total cost": m.total_cost,
                "Cost per kWh served": m.cost_per_kwh_served,
                "Diesel used (L)": m.fuel_used_l,
                "Generator hours": m.generator_hours,
                "Generator starts": m.generator_starts,
                "CO2 emissions (kg)": m.emissions_kg,
                "Renewable fraction (%)": m.renewable_fraction * 100,
                "Solar curtailed (kWh)": m.solar_curtailed_kwh,
                "Unmet energy (kWh)": m.unmet_energy_kwh,
                "LPSP (%)": m.loss_of_power_supply_probability * 100,
                "Average SoC (%)": m.average_soc * 100,
            }
            for m in metrics
        }
        table: pd.DataFrame = pd.DataFrame(rows)
        return table

    # ------------------------------------------------------------------ #
    # Heatmaps
    # ------------------------------------------------------------------ #
    def hourly_heatmap(
        self, result: SimulationResult, metric: str = "Energy demand (kWh)"
    ) -> go.Figure:
        """Day-by-hour grid of ``metric``; hover shows demand and cost of each hour."""
        column, colours = _metric(metric)
        frame = _with_derived(result.frame)
        index = pd.DatetimeIndex(frame.index)
        table = pd.DataFrame(
            {
                "date": index.strftime("%a %d %b"),
                "order": index.normalize(),
                "hour": index.hour,
                "value": frame[column].to_numpy(),
                "demand": frame["demand_kwh"].to_numpy(),
                "cost": frame["total_cost"].to_numpy(),
            }
        )

        def pivot(values: str) -> pd.DataFrame:
            grid: pd.DataFrame = table.pivot_table(
                index=["order", "date"], columns="hour", values=values, aggfunc="sum"
            )
            return grid

        value, demand, cost = pivot("value"), pivot("demand"), pivot("cost")
        figure = go.Figure(
            go.Heatmap(
                z=value.to_numpy(),
                x=[f"{hour:02d}:00" for hour in value.columns],
                y=[label for _, label in value.index],
                customdata=np.dstack([demand.to_numpy(), cost.to_numpy()]),
                colorscale=colours,
                colorbar={"title": metric},
                hovertemplate=(
                    "%{y} %{x}<br>"
                    f"{metric}: %{{z:,.2f}}<br>"
                    "Demand: %{customdata[0]:.2f} kWh<br>"
                    f"Cost: %{{customdata[1]:.2f}} {self._currency}<extra></extra>"
                ),
                xgap=1,
                ygap=1,
            )
        )
        figure.update_yaxes(autorange="reversed", title_text="Day")
        figure.update_xaxes(title_text="Hour of day")
        height = max(360, 22 * len(value.index) + 140)
        figure = self._finish(figure, f"Hourly {metric.lower()} - {result.strategy_name}", height)
        figure.update_layout(hovermode="closest")
        return figure

    def calendar_heatmap(
        self, result: SimulationResult, metric: str = "Operating cost"
    ) -> go.Figure:
        """Calendar (weeks x weekdays) of the daily total of ``metric``."""
        column, colours = _metric(metric)
        columns = list(dict.fromkeys([column, "demand_kwh", "total_cost"]))
        daily = _with_derived(result.frame)[columns].resample("D").sum()
        return self._calendar(
            values=daily[column],
            demand=daily["demand_kwh"],
            cost=daily["total_cost"],
            title=f"Daily {metric.lower()} - {result.strategy_name}",
            colour_title=metric,
            colours=colours,
        )

    def savings_calendar(
        self, baseline: SimulationResult, candidate: SimulationResult
    ) -> go.Figure:
        """Calendar of daily cost saved by ``candidate`` relative to ``baseline``."""
        columns = ["total_cost", "demand_kwh"]
        base = baseline.frame[columns].resample("D").sum()
        cand = candidate.frame[columns].resample("D").sum()
        saving = (base["total_cost"] - cand["total_cost"]).rename("saving")
        return self._calendar(
            values=saving,
            demand=cand["demand_kwh"],
            cost=cand["total_cost"],
            title=(
                f"Daily cost saved by {candidate.strategy_name} vs {baseline.strategy_name} "
                f"({self._currency})"
            ),
            colour_title=f"Saved ({self._currency})",
            colours="RdYlGn",
            centre=0.0,
        )

    def _calendar(
        self,
        *,
        values: pd.Series,
        demand: pd.Series,
        cost: pd.Series,
        title: str,
        colour_title: str,
        colours: str,
        centre: float | None = None,
    ) -> go.Figure:
        dates = pd.DatetimeIndex(values.index)
        first_monday = dates[0] - pd.Timedelta(days=int(dates[0].weekday()))
        weeks = ((dates - first_monday).days // 7).to_numpy()
        weekdays = dates.weekday.to_numpy()
        n_weeks = int(weeks.max()) + 1
        z = np.full((n_weeks, 7), np.nan)
        custom = np.full((n_weeks, 7, 3), "", dtype=object)
        text = np.full((n_weeks, 7), "", dtype=object)
        for i, stamp in enumerate(dates):
            week, day = int(weeks[i]), int(weekdays[i])
            z[week, day] = float(values.iloc[i])
            text[week, day] = str(stamp.day)
            custom[week, day] = (
                stamp.strftime("%a %d %b %Y"),
                f"{float(demand.iloc[i]):,.1f}",
                f"{float(cost.iloc[i]):,.2f}",
            )
        week_labels = [
            f"Week of {(first_monday + pd.Timedelta(weeks=w)).strftime('%d %b')}"
            for w in range(n_weeks)
        ]
        figure = go.Figure(
            go.Heatmap(
                z=z,
                x=list(_WEEKDAYS),
                y=week_labels,
                text=text,
                texttemplate="%{text}",
                customdata=custom,
                colorscale=colours,
                zmid=centre,
                colorbar={"title": colour_title},
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    f"{colour_title}: %{{z:,.2f}}<br>"
                    "Demand: %{customdata[1]} kWh<br>"
                    f"Cost: %{{customdata[2]}} {self._currency}<extra></extra>"
                ),
                xgap=4,
                ygap=4,
            )
        )
        figure.update_yaxes(autorange="reversed")
        figure.update_xaxes(side="top")
        figure = self._finish(figure, title, height=max(300, 70 * n_weeks + 150))
        figure.update_layout(hovermode="closest")
        return figure

    # ------------------------------------------------------------------ #
    # Fleet study
    # ------------------------------------------------------------------ #
    def fleet_convergence(self, report: ConsistencyReport) -> go.Figure:
        """Mean saving per fleet size with confidence bars and the reference band."""
        frame = report.to_frame()
        sizes = [str(size) for size in frame.index]
        means = frame["mean_saving_pct"].to_numpy()
        upper = (frame["ci_high_pct"] - frame["mean_saving_pct"]).fillna(0).to_numpy()
        lower = (frame["mean_saving_pct"] - frame["ci_low_pct"]).fillna(0).to_numpy()
        colours = [
            STRATEGY_COLOURS["optimizer"] if agrees else SOURCE_COLOURS["generator"]
            for agrees in frame["agrees_with_reference"]
        ]
        figure = go.Figure(
            go.Scatter(
                x=sizes,
                y=means,
                mode="lines+markers",
                name="Mean saving",
                marker={"size": 11, "color": colours},
                line={"color": STRATEGY_COLOURS["optimizer"]},
                error_y={"type": "data", "array": upper, "arrayminus": lower, "visible": True},
                hovertemplate="%{x} site(s): %{y:.1f}%<extra></extra>",
            )
        )
        ref = report.reference.savings_pct
        if ref.ci_low is not None and ref.ci_high is not None:
            figure.add_hrect(
                y0=ref.ci_low,
                y1=ref.ci_high,
                fillcolor=STRATEGY_COLOURS["optimizer"],
                opacity=0.12,
                line_width=0,
                annotation_text=f"{report.reference.size}-site 95% CI",
                annotation_position="top left",
            )
        figure.update_xaxes(title_text="Number of base stations (fleet size)", type="category")
        figure.update_yaxes(title_text=f"Cost saving vs {report.baseline} (%)")
        return self._finish(figure, f"Does the {report.candidate} saving hold as the fleet grows?")

    def fleet_distribution(self, fleet: FleetResult) -> go.Figure:
        """Per-site cost saving of every candidate strategy."""
        figure = go.Figure()
        for name in fleet.candidates:
            figure.add_trace(
                go.Box(
                    y=fleet.savings_pct(name),
                    name=name,
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    marker_color=self._strategy_colour(name),
                    text=[o.site_id for o in fleet.outcomes],
                    hovertemplate="%{text}: %{y:.1f}%<extra></extra>",
                )
            )
        figure.update_yaxes(title_text=f"Cost saving vs {fleet.baseline} (%)")
        figure = self._finish(figure, f"Per-site savings across {len(fleet)} base stations")
        figure.update_layout(hovermode="closest")
        return figure

    def fleet_drivers(self, fleet: FleetResult, candidate: str = "optimizer") -> go.Figure:
        """Saving against outage hours (marker size = average load)."""
        outcomes = fleet.outcomes
        loads = np.array([o.average_load_kw for o in outcomes])
        figure = go.Figure(
            go.Scatter(
                x=[o.outage_hours for o in outcomes],
                y=fleet.savings_pct(candidate),
                mode="markers",
                marker={
                    "size": 8 + 14 * (loads - loads.min()) / max(float(np.ptp(loads)), 1e-9),
                    "color": [o.grid_availability * 100 for o in outcomes],
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Grid availability (%)"},
                    "line": {"width": 0.5, "color": "#333"},
                },
                text=[o.site_id for o in outcomes],
                customdata=loads,
                hovertemplate=(
                    "%{text}<br>Outage hours: %{x:.0f}<br>Saving: %{y:.1f}%<br>"
                    "Avg load: %{customdata:.2f} kW<extra></extra>"
                ),
            )
        )
        figure.update_xaxes(title_text="Grid outage hours in the period")
        figure.update_yaxes(title_text=f"{candidate} cost saving (%)")
        figure = self._finish(figure, "What drives the saving? (marker size = average load)")
        figure.update_layout(hovermode="closest")
        return figure
