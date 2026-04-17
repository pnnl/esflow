import os
import pandas as pd
import numpy as np

# Define input and output paths
meta_file = "./data/sample/obs/gauge_metadata.csv"
streamflow_dir = "./data/sample/obs/streamflow"
gauge_ids = ["3629000", "4121801", "4115200", "6742900", "2969100", "1159100"]
start_date = "1985-01-01"
end_date = "1989-12-31"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_01_obs_summary/run3_output"

# Create output directory
try:
    os.makedirs(output_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating output directory {output_dir}: {e}")

# Load gauge metadata
try:
    meta_df = pd.read_csv(meta_file, dtype={"gauge_id": str})
    meta_df = meta_df[meta_df["gauge_id"].isin(gauge_ids)].set_index("gauge_id")
except Exception as e:
    print(f"Error loading metadata file {meta_file}: {e}")
    meta_df = pd.DataFrame(index=gauge_ids)

# Prepare summary list
summary_list = []

for gid in gauge_ids:
    csv_file = os.path.join(streamflow_dir, f"{gid}.csv")
    try:
        df = pd.read_csv(csv_file, parse_dates=["date"])
    except Exception as e:
        print(f"Error reading streamflow file {csv_file}: {e}")
        continue

    # Filter date range
    df = df.set_index("date").sort_index()
    df = df.loc[start_date:end_date]

    if df.empty:
        print(f"No data for gauge {gid} in period {start_date} to {end_date}")
        continue

    # Resample to monthly mean
    monthly = df["discharge_m3s"].resample("M").mean()

    # Compute summary statistics
    mean_q = monthly.mean()
    std_q = monthly.std()
    min_q = monthly.min()
    max_q = monthly.max()

    # Get metadata
    river_name = meta_df.loc[gid, "river_name"] if gid in meta_df.index and "river_name" in meta_df.columns else ""
    lat = meta_df.loc[gid, "lat"] if gid in meta_df.index and "lat" in meta_df.columns else np.nan
    lon = meta_df.loc[gid, "lon"] if gid in meta_df.index and "lon" in meta_df.columns else np.nan
    area = meta_df.loc[gid, "area_km2"] if gid in meta_df.index and "area_km2" in meta_df.columns else np.nan

    summary_list.append({
        "gauge_id": gid,
        "river_name": river_name,
        "lat": lat,
        "lon": lon,
        "area_km2": area,
        "mean_m3s": mean_q,
        "std_m3s": std_q,
        "min_m3s": min_q,
        "max_m3s": max_q
    })

# Create DataFrame and save to CSV
if summary_list:
    summary_df = pd.DataFrame(summary_list)
    out_file = os.path.join(output_dir, "obs_streamflow_summary.csv")
    try:
        summary_df.to_csv(out_file, index=False)
        print(f"Summary saved to {out_file}")
    except Exception as e:
        print(f"Error saving summary CSV {out_file}: {e}")
else:
    print("No summary data to save.")