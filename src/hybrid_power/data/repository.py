"""Persistence of operating profiles as CSV files."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from typing_extensions import override

from hybrid_power.data.profiles import ProfileProvider, ProfileSet
from hybrid_power.exceptions import ProfileDataError


class CsvProfileRepository(ProfileProvider):
    """Load and save :class:`ProfileSet` objects as CSV.

    The repository can read from a filesystem path or from an in-memory
    buffer (used by the dashboard's file uploader).
    """

    def __init__(self, source: Path | io.StringIO | io.BytesIO) -> None:
        self._source = source

    @property
    def source(self) -> Path | io.StringIO | io.BytesIO:
        """Where the profile is read from."""
        return self._source

    @override
    def get_profile(self) -> ProfileSet:
        try:
            frame = pd.read_csv(self._source)
        except FileNotFoundError as exc:
            raise ProfileDataError(f"Profile file not found: {self._source}") from exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise ProfileDataError(f"Could not parse profile CSV: {exc}") from exc
        return ProfileSet.from_frame(frame)

    @staticmethod
    def save(profile: ProfileSet, path: Path) -> Path:
        """Write ``profile`` to ``path`` (creating parent folders) and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        profile.to_frame().to_csv(path, index=False, float_format="%.5f")
        return path
