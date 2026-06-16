#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    # Input paths
    metadata_path = "./data/sample/obs/gauge_metadata.csv"
    streamflow_dir = "./data/sample/obs/streamflow"

    # Output directory (use the absolute path specified by user)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_01_obs_summary/run4_output"

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create output directory {output_dir}: {e}", file=sys.stderr)
        return

    # Gauge IDs of interest with names for reference
    gauges_info = {
        3629000: "Amazon",
        4121801: "Missouri",
        4115200: "Columbia",
        6742900: "Danube",
        2969100: "Mekong",
        1159100: "Orange"
    }

    start_date = "1985-01-01"
    end_date = "1989-12-31"

    # Load gauge metadata
    try:
        meta = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"ERROR: Failed to read metadata file {metadata_path}: {e}", file=sys.stderr)
        return

    # Normalize metadata types and columns
    # Ensure gauge_id column exists
    if 'gauge_id' not in meta.columns:
        print("ERROR: 'gauge_id' column not found in metadata.", file=sys.stderr)
        return

    # Convert gauge_id to int where possible
    try:
        meta['gauge_id'] = pd.to_numeric(meta['gauge_id'], errors='coerce').astype('Int64')
    except Exception as e:
        print(f"WARNING: Could not convert gauge_id to numeric: {e}", file=sys.stderr)

    # Prepare containers
    monthly_list = []
    summary_rows = []

    for gid, gname in gauges_info.items():
        # Select metadata row for this gauge
        mrow = meta.loc[meta['gauge_id'] == gid]
        if mrow.empty:
            print(f"WARNING: Gauge ID {gid} ({gname}) not found in metadata. Proceeding with limited info.", file=sys.stderr)
            lat = np.nan
            lon = np.nan
            area_km2 = np.nan
            river_name = gname  # fallback to given label
        else:
            # Take first matching row
            row = mrow.iloc[0]
            lat = row['lat'] if 'lat' in mrow.columns else np.nan
            lon = row['lon'] if 'lon' in mrow.columns else np.nan
            area_km2 = row['area_km2'] if 'area_km2' in mrow.columns else np.nan
            river_name = row['river_name'] if 'river_name' in mrow.columns else gname

        # Read per-gauge streamflow CSV
        gauge_csv = os.path.join(streamflow_dir, f"{gid}.csv")
        try:
            df = pd.read_csv(gauge_csv)
        except Exception as e:
            print(f"ERROR: Failed to read streamflow file for gauge {gid} at {gauge_csv}: {e}", file=sys.stderr)
            continue

        # Ensure necessary columns exist
        if 'date' not in df.columns or 'discharge_m3s' not in df.columns:
            print(f"ERROR: Missing required columns in {gauge_csv}. Expected 'date' and 'discharge_m3s'.", file=sys.stderr)
            continue

        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.set_index('date').sort_index()

        # Filter to desired period
        df_period = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))].copy()
        if df_period.empty:
            print(f"WARNING: No data for gauge {gid} in period {start_date} to {end_date}.", file=sys.stderr)

        # Resample to monthly means
        monthly = df_period['discharge_m3s'].resample('MS').mean().to_frame()
        monthly = monthly.rename(columns={'discharge_m3s': 'discharge_m3s_monthly_mean'})

        # Add metadata columns
        monthly.reset_index(inplace=True)
        monthly['gauge_id'] = gid
        monthly['gauge_name'] = gname
        monthly['river_name'] = river_name
        monthly['lat'] = lat
        monthly['lon'] = lon
        monthly['area_km2'] = area_km2

        # Reorder columns
        monthly = monthly[['gauge_id', 'gauge_name', 'river_name', 'lat', 'lon', 'area_km2', 'date', 'discharge_m3s_monthly_mean']]

        monthly_list.append(monthly)

        # Compute summary statistics over the period
        values = monthly['discharge_m3s_monthly_mean'].dropna()
        if values.empty:
            mean_v = np.nan
            std_v = np.nan
            min_v = np.nan
            max_v = np.nan
            count_v = 0
        else:
            mean_v = values.mean()
            std_v = values.std(ddof=1) if len(values) > 1 else 0.0
            min_v = values.min()
            max_v = values.max()
            count_v = values.shape[0]

        summary_rows.append({
            'gauge_id': gid,
            'gauge_name': gname,
            'river_name': river_name,
            'lat': lat,
            'lon': lon,
            'area_km2': area_km2,
            'period_start': start_date,
            'period_end': end_date,
            'n_months': count_v,
            'mean_discharge_m3s': mean_v,
            'std_discharge_m3s': std_v,
            'min_discharge_m3s': min_v,
            'max_discharge_m3s': max_v
        })

    # Combine monthly results and save
    if monthly_list:
        monthly_all = pd.concat(monthly_list, ignore_index=True)
        monthly_csv_path = os.path.join(output_dir, "monthly_streamflow_observations_1985_1989.csv")
        try:
            monthly_all.to_csv(monthly_csv_path, index=False)
            print(f"Saved monthly streamflow observations to {monthly_csv_path}")
        except Exception as e:
            print(f"ERROR: Failed to save monthly observations CSV: {e}", file=sys.stderr)
    else:
        print("WARNING: No monthly data compiled for any gauge.", file=sys.stderr)

    # Save summary statistics
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "obs_streamflow_summary_1985_1989.csv")
    try:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary statistics to {summary_csv_path}")
    except Exception as e:
        print(f"ERROR: Failed to save summary statistics CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()