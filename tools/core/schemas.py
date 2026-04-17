"""
Standard intermediate data schemas for ESMFlow tools (documentation reference).

These schemas describe the CSV contracts between tools. They are not serialized
to the catalog — the catalog uses ToolSpec.outputs directly. This file exists
as a human-readable reference for tool developers.
"""

SCHEMAS = {
    "gauge_metadata": "gauge_id, lat, lon, area_km2, river_name",
    "matched_gauges": "gauge_id, lat, lon, model_lat, model_lon, lat_idx, lon_idx",
    "timeseries": "time (index), one column per gauge_id (float values, m3/s)",
    "climatology": "month (index 1-12), one column per gauge_id",
    "metrics": "gauge_id, nse, kge, pbias, rmse, correlation, n_valid",
}
