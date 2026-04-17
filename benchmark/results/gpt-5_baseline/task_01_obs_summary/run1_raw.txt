import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    # Input paths
    gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"
    streamflow_dir = "./data/sample/obs/streamflow"

    # Output directory (use the absolute path specified by the user)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_01_obs_summary/run1_output"
    os.makedirs(output_dir, exist_ok=True)

    # Period
    start_date = pd.Timestamp("1985-01-01")
    end_date = pd.Timestamp("1989-12-31")

    # Gauges of interest: river names and their IDs
    gauges = {
        "Amazon": "3629000",
        "Missouri": "4121801",
        "Columbia": "4115200",
        "Danube": "6742900",
        "Mekong": "2969100",
        "Orange": "1159100",
    }

    # Load gauge metadata
    try:
        meta = pd.read_csv(gauge_metadata_path, dtype={"gauge_id": str})
    except Exception as e:
        print(f"Error reading gauge metadata: {e}", file=sys.stderr)
        meta = pd.DataFrame(columns=["gauge_id", "lat", "lon", "area_km2", "river_name"])

    # Prepare containers for results
    stats_records = []
    monthly_ts_records = []

    for river_name, gauge_id in gauges.items():
        csv_path = os.path.join(streamflow_dir, f"{gauge_id}.csv")
        print(f"Processing gauge {gauge_id} ({river_name}) from {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Error reading {csv_path}: {e}", file=sys.stderr)
            continue

        # Ensure expected columns exist
        if "date" not in df.columns or "discharge_m3s" not in df.columns:
            print(f"  Missing required columns in {csv_path}", file=sys.stderr)
            continue

        # Parse dates and clean
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            print(f"  Error parsing dates in {csv_path}: {e}", file=sys.stderr)
            continue

        # Keep within period and set index
        df = df.sort_values("date")
        df = df[(df["date"] >= start_date) and (df["date"] <= end_date)] if False else df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            print(f"  No data within the period for {gauge_id}", file=sys.stderr)
            continue

        df = df.set_index("date")
        # Coerce to numeric
        df["discharge_m3s"] = pd.to_numeric(df["discharge_m3s"], errors="coerce")

        # Resample to monthly means (end of month label)
        monthly = df["discharge_m3s"].resample("M").mean().dropna()

        if monthly.empty:
            print(f"  Monthly time series is empty after resampling for {gauge_id}", file=sys.stderr)
            continue

        # Compute statistics over 1985-1989 monthly series
        stats = {
            "gauge_id": gauge_id,
            "river_name_requested": river_name,
            "period_start": start_date.date().isoformat(),
            "period_end": end_date.date().isoformat(),
            "n_months": int(monthly.shape[0]),
            "mean_m3s": float(monthly.mean()),
            "std_m3s": float(monthly.std(ddof=1)) if monthly.shape[0] > 1 else np.nan,
            "min_m3s": float(monthly.min()),
            "max_m3s": float(monthly.max()),
        }

        # Attach metadata if available
        row_meta = meta[meta["gauge_id"] == gauge_id]
        if not row_meta.empty:
            r = row_meta.iloc[0]
            stats.update({
                "lat": float(r["lat"]) if "lat" in r and pd.notna(r["lat"]) else np.nan,
                "lon": float(r["lon"]) if "lon" in r and pd.notna(r["lon"]) else np.nan,
                "area_km2": float(r["area_km2"]) if "area_km2" in r and pd.notna(r["area_km2"]) else np.nan,
                "river_name_meta": str(r["river_name"]) if "river_name" in r and pd.notna(r["river_name"]) else "",
            })
        else:
            stats.update({"lat": np.nan, "lon": np.nan, "area_km2": np.nan, "river_name_meta": ""})

        stats_records.append(stats)

        # Store monthly time series records
        for ts_date, q in monthly.items():
            monthly_ts_records.append({
                "gauge_id": gauge_id,
                "river_name_requested": river_name,
                "date": ts_date.date().isoformat(),
                "discharge_m3s": float(q),
            })

    # Create DataFrames
    stats_df = pd.DataFrame(stats_records)
    monthly_ts_df = pd.DataFrame(monthly_ts_records)

    # Sort outputs
    if not stats_df.empty:
        stats_df = stats_df.sort_values(["river_name_requested", "gauge_id"]).reset_index(drop=True)
    if not monthly_ts_df.empty:
        # Ensure proper ordering by date and river
        monthly_ts_df["date"] = pd.to_datetime(monthly_ts_df["date"])
        monthly_ts_df = monthly_ts_df.sort_values(["river_name_requested", "gauge_id", "date"]).reset_index(drop=True)
        # Convert date back to ISO strings for CSV
        monthly_ts_df["date"] = monthly_ts_df["date"].dt.date.astype(str)

    # Save outputs
    stats_csv = os.path.join(output_dir, "obs_streamflow_monthly_stats_1985_1989.csv")
    ts_csv = os.path.join(output_dir, "obs_streamflow_monthly_timeseries_1985_1989.csv")

    try:
        if not stats_df.empty:
            stats_df.to_csv(stats_csv, index=False)
            print(f"Saved summary statistics to {stats_csv}")
        else:
            print("No statistics to save; all gauges failed or had no data within period.", file=sys.stderr)
    except Exception as e:
        print(f"Error saving statistics CSV: {e}", file=sys.stderr)

    try:
        if not monthly_ts_df.empty:
            monthly_ts_df.to_csv(ts_csv, index=False)
            print(f"Saved monthly time series to {ts_csv}")
        else:
            print("No monthly time series to save.", file=sys.stderr)
    except Exception as e:
        print(f"Error saving monthly time series CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()