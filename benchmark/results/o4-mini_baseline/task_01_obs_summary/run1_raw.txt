import os
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    # List of gauges to process
    gauge_ids = ['3629000', '4121801', '4115200', '6742900', '2969100', '1159100']
    # Paths
    metadata_path = "./data/sample/obs/gauge_metadata.csv"
    streamflow_dir = "./data/sample/obs/streamflow"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_01_obs_summary/run1_output"
    summary_csv = os.path.join(output_dir, "obs_summary.csv")

    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return

    # Read metadata
    try:
        meta_df = pd.read_csv(metadata_path, dtype={"gauge_id": str})
    except Exception as e:
        print(f"Error reading metadata file {metadata_path}: {e}")
        return

    # Filter metadata for selected gauges
    meta_df = meta_df[meta_df["gauge_id"].isin(gauge_ids)]
    meta_dict = meta_df.set_index("gauge_id").to_dict(orient="index")

    # Prepare list for summary
    summary_list = []

    # Define period
    start_date = "1985-01-01"
    end_date = "1989-12-31"

    # Process each gauge
    for gid in gauge_ids:
        fname = os.path.join(streamflow_dir, f"{gid}.csv")
        try:
            df = pd.read_csv(fname, parse_dates=["date"])
        except Exception as e:
            print(f"Error reading streamflow file {fname}: {e}")
            continue

        if "discharge_m3s" not in df.columns:
            print(f"File {fname} missing 'discharge_m3s' column")
            continue

        # Filter period
        df = df.set_index("date")
        df = df.loc[start_date:end_date]

        # Resample to monthly mean
        monthly = df["discharge_m3s"].resample("M").mean()

        # Drop missing
        monthly = monthly.dropna()
        if monthly.empty:
            print(f"No data for gauge {gid} in period {start_date} to {end_date}")
            continue

        # Compute statistics
        g_mean = float(monthly.mean())
        g_std = float(monthly.std())
        g_min = float(monthly.min())
        g_max = float(monthly.max())

        # Get metadata
        river_name = meta_dict.get(gid, {}).get("river_name", "")

        summary_list.append({
            "gauge_id": gid,
            "river_name": river_name,
            "mean_m3s": g_mean,
            "std_m3s": g_std,
            "min_m3s": g_min,
            "max_m3s": g_max
        })

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_list,
                              columns=["gauge_id", "river_name", "mean_m3s", "std_m3s", "min_m3s", "max_m3s"])
    # Save to CSV
    try:
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary saved to {summary_csv}")
    except Exception as e:
        print(f"Error writing summary CSV {summary_csv}: {e}")

if __name__ == "__main__":
    main()