import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    # Input paths
    metadata_csv = "./data/sample/obs/gauge_metadata.csv"
    obs_stream_dir = "./data/sample/obs/streamflow"

    # Output directory (as specified by user)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_01_obs_summary/run3_output"
    os.makedirs(output_dir, exist_ok=True)

    # Gauge selection (name: gauge_id)
    gauges = {
        "Amazon": 3629000,
        "Missouri": 4121801,
        "Columbia": 4115200,
        "Danube": 6742900,
        "Mekong": 2969100,
        "Orange": 1159100,
    }

    start_date = pd.Timestamp("1985-01-01")
    end_date = pd.Timestamp("1989-12-31")

    # Load gauge metadata
    metadata = None
    try:
        metadata = pd.read_csv(metadata_csv)
    except Exception as e:
        print(f"Error reading metadata file '{metadata_csv}': {e}", file=sys.stderr)
        metadata = pd.DataFrame(columns=["gauge_id", "lat", "lon", "area_km2", "river_name"])

    # Ensure types for joining
    if "gauge_id" in metadata.columns:
        # Cast to int where possible
        try:
            metadata["gauge_id"] = metadata["gauge_id"].astype(int)
        except Exception:
            # Fallback: coerce to numeric then int
            metadata["gauge_id"] = pd.to_numeric(metadata["gauge_id"], errors="coerce").astype("Int64")

    # Prepare containers
    summary_rows = []
    monthly_series_dict = {}

    for name, gid in gauges.items():
        gauge_file = os.path.join(obs_stream_dir, f"{gid}.csv")
        print(f"Processing gauge {name} (ID: {gid}) from {gauge_file}")
        try:
            df = pd.read_csv(gauge_file, parse_dates=["date"])
        except Exception as e:
            print(f"  Warning: Unable to read {gauge_file}: {e}", file=sys.stderr)
            continue

        # Validate required columns
        if "date" not in df.columns or "discharge_m3s" not in df.columns:
            print(f"  Warning: Missing required columns in {gauge_file}. Expected 'date' and 'discharge_m3s'.", file=sys.stderr)
            continue

        # Clean and filter date range
        try:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date").sort_index()
            df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        except Exception as e:
            print(f"  Warning: Error processing dates for {gauge_file}: {e}", file=sys.stderr)
            continue

        if df.empty:
            print(f"  Warning: No data within {start_date.date()} to {end_date.date()} for {gauge_file}", file=sys.stderr)
            continue

        # Ensure discharge is numeric
        df["discharge_m3s"] = pd.to_numeric(df["discharge_m3s"], errors="coerce")
        # Resample to monthly means (label at month start)
        monthly = df["discharge_m3s"].resample("MS").mean()

        # Store monthly series
        monthly_series_dict[name] = monthly

        # Compute summary stats over monthly values
        valid_monthly = monthly.dropna()
        mean_val = valid_monthly.mean() if not valid_monthly.empty else np.nan
        std_val = valid_monthly.std(ddof=1) if len(valid_monthly) > 1 else np.nan
        min_val = valid_monthly.min() if not valid_monthly.empty else np.nan
        max_val = valid_monthly.max() if not valid_monthly.empty else np.nan
        count_months = valid_monthly.shape[0]

        # Merge with metadata for context
        meta_row = {}
        if metadata is not None and not metadata.empty and "gauge_id" in metadata.columns:
            meta_match = metadata[metadata["gauge_id"] == gid]
            if not meta_match.empty:
                meta_row = {
                    "lat": meta_match.iloc[0].get("lat", np.nan),
                    "lon": meta_match.iloc[0].get("lon", np.nan),
                    "area_km2": meta_match.iloc[0].get("area_km2", np.nan),
                    "river_name": meta_match.iloc[0].get("river_name", ""),
                }
            else:
                meta_row = {"lat": np.nan, "lon": np.nan, "area_km2": np.nan, "river_name": ""}
        else:
            meta_row = {"lat": np.nan, "lon": np.nan, "area_km2": np.nan, "river_name": ""}

        summary_rows.append({
            "gauge_name": name,
            "gauge_id": gid,
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "months_with_data": count_months,
            "mean_discharge_m3s": mean_val,
            "std_discharge_m3s": std_val,
            "min_discharge_m3s": min_val,
            "max_discharge_m3s": max_val,
            **meta_row
        })

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    # Sort by gauge name for readability
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=["gauge_name"]).reset_index(drop=True)

    # Save summary CSV
    summary_csv_path = os.path.join(output_dir, "streamflow_monthly_summary_1985_1989.csv")
    try:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary statistics to: {summary_csv_path}")
    except Exception as e:
        print(f"Error saving summary CSV to {summary_csv_path}: {e}", file=sys.stderr)

    # Also save monthly time series for reference (one file, columns per gauge)
    if monthly_series_dict:
        # Align all monthly series on a common index
        all_months = pd.date_range(start=start_date, end=end_date, freq="MS")
        monthly_df = pd.DataFrame(index=all_months)
        for name, ser in monthly_series_dict.items():
            monthly_df[name] = ser.reindex(all_months)
        monthly_series_csv_path = os.path.join(output_dir, "streamflow_monthly_series_1985_1989.csv")
        try:
            monthly_df.to_csv(monthly_series_csv_path, index_label="month_start")
            print(f"Saved monthly series to: {monthly_series_csv_path}")
        except Exception as e:
            print(f"Error saving monthly series CSV to {monthly_series_csv_path}: {e}", file=sys.stderr)
    else:
        print("No monthly series to save.", file=sys.stderr)

if __name__ == "__main__":
    main()