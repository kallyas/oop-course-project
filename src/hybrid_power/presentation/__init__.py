"""Presentation layer: chart factory and Streamlit dashboard.

The dashboard module is not imported here so that the chart factory (and the
rest of the package) can be used without importing Streamlit.
"""

from hybrid_power.presentation.charts import ChartFactory

__all__ = ["ChartFactory"]
