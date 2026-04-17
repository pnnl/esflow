import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    # Input paths
    gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"
    streamflow_dir = "./data/sample/obs/streamflow"

    # Output directory (use the explicit path provided by the user)
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gpt-5_baseline/task_01_obs_summary/run2_output"

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
        return

    # Gauge IDs and river names to process
    gauges = [
        {"river_name": "Amazon", "gauge_id": 3629000},
        {"river_name": "Missouri", "gauge_id": 4121801},
        {"river_name": "Columbia", "gauge_id": 4115200},
        {"river_name": "Danube", "gauge_id": 6742900},
        {"river_name": "Mekong", "gauge_id": 2969100},
        {"river_name": "Orange", "gauge_id": 1159100},
    ]

    # Load gauge metadata
    try:
        meta = pd.read_csv(gauge_metadata_path)
    except Exception as e:
        print(f"Error reading gauge metadata from {gauge_metadata_path}: {e}", file=sys.stderr)
        return

    # Coerce gauge_id types for merging
    # Ensure metadata gauge_id is integer if possible
    if 'gauge_id' in meta.columns:
        try:
            meta['gauge_id'] = meta['gauge_id'].astype(int)
        except Exception:
            # If not convertible, leave as is and handle during merge
            pass
    else:
        print(f"Metadata file {gauge_metadata_path} missing 'gauge_id' column.", file=sys.stderr)
        return

    # Filter metadata to the gauges of interest
    gauge_ids = [g["gauge_id"] for g in gauges]
    meta_sel = meta[meta['gauge_id'].isin(gauge_ids)].copy()

    # Create mapping from gauge_id to metadata columns for easy lookup
    meta_map = {}
    for _, row in meta_sel.iterrows():
        meta_map[int(row['gauge_id'])] = {
            "lat": row.get('lat', np.nan),
            "lon": row.get('lon', np.nan),
            "area_km2": row.get('area_km2', np.nan),
            "river_name_meta": row.get('river_name', None),
        }

    # Analysis period
    start_date = pd.Timestamp("1985-01-01")
    end_date = pd.Timestamp("1989-12-31")

    results = []

    for g in gauges:
        gauge_id = g["gauge_id"]
        river_name_requested = g["river_name"]

        # Get metadata if available
        lat = np.nan
        lon = np.nan
        area_km2 = np.nan
        river_name_meta = None
        if gauge_id in meta_map:
            lat = meta_map[gauge_id]["lat"]
            lon = meta_map[gauge_id]["lon"]
            area_km2 = meta_map[gauge_id]["area_km2"]
            river_name_meta = meta_map[gauge_id]["river_name_meta"]

        # Read per-gauge streamflow CSV
        gauge_csv = os.path.join(streamflow_dir, f"{gauge_id}.csv")
        try:
            df = pd.read_csv(gauge_csv, parse_dates=['date'])
        except Exception as e:
            print(f"Error reading streamflow for gauge {gauge_id} from {gauge_csv}: {e}", file=sys.stderr)
            # Record NaN stats if file missing or unreadable
            results.append({
                "gauge_id": gauge_id,
                "river_name_requested": river_name_requested,
                "river_name_meta": river_name_meta,
                "lat": lat,
                "lon": lon,
                "area_km2": area_km2,
                "period_start": start_date.date(),
                "period_end": end_date.date(),
                "n_months": 0,
                "mean_m3s": np.nan,
                "std_m3s": np.nan,
                "min_m3s": np.nan,
                "max_m3s": np.nan,
            })
            continue

        # Ensure expected columns exist
        if 'date' not in df.columns or 'discharge_m3s' not in df.columns:
            print(f"Gauge file {gauge_csv} missing required columns 'date' and/or 'discharge_m3s'.", file=sys.stderr)
            results.append({
                "gauge_id": gauge_id,
                "river_name_requested": river_name_requested,
                "river_name_meta": river_name_meta,
                "lat": lat,
                "lon": lon,
                "area_km2": area_km2,
                "period_start": start_date.date(),
                "period_end": end_date.date(),
                "n_months": 0,
                "mean_m3s": np.nan,
                "std_m3s": np.nan,
                "min_m3s": np.nan,
                "max_m3s": np.nan,
            })
            continue

        # Prepare time series
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
        except Exception as e:
            print(f"Error processing dates for gauge {gauge_id}: {e}", file=sys.stderr)
            results.append({
                "gauge_id": gauge_id,
                "river_name_requested": river_name_requested,
                "river_name_meta": river_name_meta,
                "lat": lat,
                "lon": lon,
                "area_km2": area_km2,
                "period_start": start_date.date(),
                "period_end": end_date.date(),
                "n_months": 0,
                "mean_m3s": np.nan,
                "std_m3s": np.nan,
                "min_m3s": np.nan,
                "max_m3s": np.nan,
            })
            continue

        # Filter period
        df_period = df.loc[(df.index >= start_date) & (df.index <= end_date)]
        if df_period.empty:
            print(f"No data for gauge {gauge_id} in the period {start_date.date()} to {end_date.date()}.", file=sys.stderr)
            results.append({
                "gauge_id": gauge_id,
                "river_name_requested": river_name_requested,
                "river_name_meta": river_name_meta,
                "lat": lat,
                "lon": lon,
                "area_km2": area_km2,
                "period_start": start_date.date(),
                "period_end": end_date.date(),
                "n_months": 0,
                "mean_m3s": np.nan,
                "std_m3s": np.nan,
                "min_m3s": np.nan,
                "max_m3s": np.nan,
            })
            continue

        # Resample to monthly means (discharge in m3/s)
        try:
            monthly = df_period['discharge_m3s'].resample('M').mean()
        except Exception as e:
            print(f"Error resampling to monthly for gauge {gauge_id}: {e}", file=sys.stderr)
            results.append({
                "gauge_id": gauge_id,
                "river_name_requested": river_name_requested,
                "river_name_meta": river_name_meta,
                "lat": lat,
                "lon": lon,
                "area_km2": area_km2,
                "period_start": start_date.date(),
                "period_end": end_date.date(),
                "n_months": 0,
                "mean_m3s": np.nan,
                "std_m3s": np.nan,
                "min_m3s": np.nan,
                "max_m3s": np.nan,
            })
            continue

        # Drop months that are all NaN (if any)
        monthly = monthly.dropna()

        # Compute summary statistics over monthly means
        mean_val = monthly.mean() if not monthly.empty else np.nan
        std_val = monthly.std(ddof=1) if len(monthly) > 1 else (0.0 if len(monthly) == 1 else np.nan)
        min_val = monthly.min() if not monthly.empty else np.nan
        max_val = monthly.max() if not monthly.empty else np.nan

        results.append({
            "gauge_id": gauge_id,
            "river_name_requested": river_name_requested,
            "river_name_meta": river_name_meta,
            "lat": lat,
            "lon": lon,
            "area_km2": area_km2,
            "period_start": start_date.date(),
            "period_end": end_date.date(),
            "n_months": int(len(monthly)),
            "mean_m3s": float(mean_val) if pd.notnull(mean_val) else np.nan,
            "std_m3s": float(std_val) if pd.notnull(std_val) else np.nan,
            "min_m3s": float(min_val) if pd.notnull(min_val) else np.nan,
            "max_m3s": float(max_val) if pd.notnull(max_val) else np.nan,
        })

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    # Sort by river name for readability
    results_df = results_df.sort_values(by=['river_name_requested', 'gauge_id'])

    out_csv = os.path.join(output_dir, "obs_streamflow_summary_1985_1989.csv")
    try:
        results_df.to_csv(out_csv, index=False)
        print(f"Saved summary statistics to {out_csv}")
    except Exception as e:
        print(f"Error writing summary CSV to {out_csv}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()