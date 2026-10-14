"""Command-line interface.

Examples::

    hybrid-power simulate --days 30 --seed 7
    hybrid-power simulate --config data/sample_scenario.json --export-dir results/
    hybrid-power fleet --sites 50 --export-dir results/
    hybrid-power generate-profile --output data/sample_profile.csv
    hybrid-power dashboard
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from hybrid_power import __version__
from hybrid_power.config import FleetConfig, ScenarioConfig
from hybrid_power.data.repository import CsvProfileRepository
from hybrid_power.data.synthetic import SyntheticProfileGenerator
from hybrid_power.exceptions import HybridPowerError
from hybrid_power.fleet import FleetStudyOutcome, run_fleet_study
from hybrid_power.service import SimulationOutcome, SimulationService
from hybrid_power.simulation.engine import LoggingObserver

EXIT_OK = 0
EXIT_ERROR = 2


class CommandLineInterface:
    """Parse arguments and dispatch to sub-command handlers."""

    def __init__(
        self,
        service: SimulationService | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self._service = service or SimulationService()
        self._out = stdout or sys.stdout
        self._err = stderr or sys.stderr
        self._parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="hybrid-power",
            description="Hybrid power optimisation simulator for telecom base stations.",
        )
        parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
        parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
        commands = parser.add_subparsers(dest="command", required=True)

        simulate = commands.add_parser("simulate", help="run and compare dispatch strategies")
        self._add_scenario_arguments(simulate)
        simulate.add_argument("--profile", type=Path, help="CSV operating profile to use")
        simulate.add_argument(
            "--strategy",
            action="append",
            choices=self._service.registry.keys(),
            help="strategy to run (repeatable; default: all)",
        )
        simulate.add_argument("--export-dir", type=Path, help="write CSV results to this folder")
        simulate.set_defaults(handler=self._simulate)

        fleet = commands.add_parser(
            "fleet", help="check whether results are consistent from few to many base stations"
        )
        self._add_scenario_arguments(fleet)
        fleet.add_argument("--sites", type=int, default=50, help="largest fleet size (default 50)")
        fleet.add_argument(
            "--sizes",
            type=int,
            nargs="+",
            help="fleet sizes to compare (default: 1 5 10 25 ... up to --sites)",
        )
        fleet.add_argument("--fleet-seed", type=int, default=2026, help="site sampling seed")
        fleet.add_argument("--export-dir", type=Path, help="write fleet CSV results here")
        fleet.set_defaults(handler=self._fleet)

        generate = commands.add_parser("generate-profile", help="write a synthetic profile CSV")
        self._add_scenario_arguments(generate)
        generate.add_argument("--output", type=Path, required=True, help="CSV file to create")
        generate.set_defaults(handler=self._generate_profile)

        dashboard = commands.add_parser("dashboard", help="launch the Streamlit dashboard")
        dashboard.set_defaults(handler=self._dashboard)
        return parser

    @staticmethod
    def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", type=Path, help="scenario JSON file")
        parser.add_argument("--days", type=int, help="override the number of simulated days")
        parser.add_argument("--seed", type=int, help="override the random seed")

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def run(self, argv: Sequence[str] | None = None) -> int:
        """Execute the CLI and return a process exit code."""
        args = self._parser.parse_args(argv)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
        )
        try:
            code: int = args.handler(args)
        except HybridPowerError as exc:
            self._err.write(f"error: {exc}\n")
            return EXIT_ERROR
        return code

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scenario(args: argparse.Namespace) -> ScenarioConfig:
        config = ScenarioConfig.load(args.config) if args.config else ScenarioConfig()
        overrides: dict[str, int] = {}
        if args.days is not None:
            overrides["days"] = args.days
        if args.seed is not None:
            overrides["seed"] = args.seed
        if overrides:
            profile = type(config.profile).from_dict({**config.profile.to_dict(), **overrides})
            config = replace(config, profile=profile)
        return config

    def _simulate(self, args: argparse.Namespace) -> int:
        config = self._scenario(args)
        provider = CsvProfileRepository(args.profile) if args.profile else None
        outcome = self._service.run(
            config,
            provider=provider,
            strategies=args.strategy,
            observers=[LoggingObserver()] if args.verbose else (),
        )
        self._print_summary(outcome)
        if args.export_dir:
            self._export(outcome, args.export_dir)
        return EXIT_OK

    def _fleet(self, args: argparse.Namespace) -> int:
        template = self._scenario(args)
        if args.sizes:
            fleet_config = FleetConfig(
                sites=max(args.sizes), fleet_sizes=tuple(args.sizes), seed=args.fleet_seed
            )
        else:
            fleet_config = FleetConfig.with_sites(args.sites, seed=args.fleet_seed)
        self._out.write(
            f"Simulating {fleet_config.sites} base stations x "
            f"{len(self._service.registry)} strategies...\n"
        )
        study = run_fleet_study(template, fleet_config)
        self._print_fleet(study)
        if args.export_dir:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            study.fleet.to_frame().to_csv(args.export_dir / "fleet_sites.csv", index=False)
            study.report.to_frame().to_csv(args.export_dir / "fleet_consistency.csv")
            self._out.write(f"Fleet results exported to {args.export_dir}\n")
        return EXIT_OK

    def _print_fleet(self, study: FleetStudyOutcome) -> None:
        table = study.report.to_frame()[
            ["mean_saving_pct", "ci_low_pct", "ci_high_pct", "win_rate", "recommended"]
        ].rename(
            columns={
                "mean_saving_pct": "mean saving %",
                "ci_low_pct": "CI low %",
                "ci_high_pct": "CI high %",
                "win_rate": "win rate",
            }
        )
        self._out.write("\n" + table.to_string(float_format=lambda v: f"{v:,.2f}", na_rep="-"))
        self._out.write("\n\n" + study.report.verdict() + "\n")

    def _generate_profile(self, args: argparse.Namespace) -> int:
        config = self._scenario(args)
        profile = SyntheticProfileGenerator(config.profile).get_profile()
        path = CsvProfileRepository.save(profile, args.output)
        self._out.write(f"Wrote {len(profile)} steps to {path}\n")
        return EXIT_OK

    def _dashboard(self, _args: argparse.Namespace) -> int:
        app = Path(__file__).with_name("presentation") / "dashboard_entry.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def _print_summary(self, outcome: SimulationOutcome) -> None:
        report = outcome.report
        profile = outcome.profile
        currency = outcome.config.currency
        self._out.write(
            f"Scenario: {outcome.config.name} | {len(profile)} steps | "
            f"grid availability {profile.grid_availability_ratio:.1%}\n\n"
        )
        table = report.to_frame()[
            [
                "total_cost",
                "fuel_used_l",
                "generator_hours",
                "emissions_kg",
                "renewable_fraction",
                "unmet_energy_kwh",
            ]
        ].rename(
            columns={
                "total_cost": f"cost ({currency})",
                "fuel_used_l": "diesel (L)",
                "generator_hours": "gen hours",
                "emissions_kg": "CO2 (kg)",
                "renewable_fraction": "renewable",
                "unmet_energy_kwh": "unmet (kWh)",
            }
        )
        self._out.write(table.to_string(float_format=lambda v: f"{v:,.2f}") + "\n\n")
        for name, comparison in report.comparisons().items():
            pct = comparison.cost_savings_pct
            pct_text = f" ({pct:.1f}%)" if pct is not None else ""
            self._out.write(
                f"{name}: saves {comparison.cost_savings:,.2f} {currency}{pct_text}, "
                f"{comparison.fuel_savings_l:,.1f} L diesel, "
                f"{comparison.emissions_reduction_kg:,.1f} kg CO2 vs {report.baseline_name}\n"
            )
        self._out.write(f"\nRecommended strategy: {report.best().strategy_name}\n")

    def _export(self, outcome: SimulationOutcome, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        outcome.report.to_frame().to_csv(directory / "metrics.csv")
        CsvProfileRepository.save(outcome.profile, directory / "profile.csv")
        outcome.config.save(directory / "scenario.json")
        for name, result in outcome.results.items():
            result.frame.to_csv(directory / f"results_{name}.csv")
        self._out.write(f"Results exported to {directory}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    return CommandLineInterface().run(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
