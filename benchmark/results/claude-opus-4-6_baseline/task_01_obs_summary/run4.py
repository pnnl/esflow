import os
import pandas as pd
import numpy as np

output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_01_obs_summary/run4_output"
os.makedirs(output_dir, exist_ok=True)

# Define the six gauges of interest
gauges = {
    "3629000": "Amazon",
    "4121801": "Missouri",
    "4115200": "Columbia",
    "6742900": "Danube",
    "2969100": "Mekong",
    "1159100": "Orange",
}

# Load gauge metadata
try:
    metadata = pd.read_csv("./data/sample/obs/gauge_metadata.csv")
    print("Gauge metadata loaded successfully.")
    print(metadata)
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    metadata = None

# Load daily streamflow observations for each gauge and compute monthly means
start_date = "1985-01-01"
end_date = "1989-12-31"

monthly_data = {}
daily_data = {}

for gauge_id, river_name in gauges.items():
    filepath = f"./data/sample/obs/streamflow/{gauge_id}.csv"
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        print(f"Loaded data for {river_name} (gauge {gauge_id}): {len(df)} records")

        # Filter to 1985-1989
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df_period = df.loc[mask].copy()
        print(f"  Records in 1985-1989: {len(df_period)}")

        if len(df_period) == 0:
            print(f"  WARNING: No data in the 1985-1989 period for {river_name}")
            continue

        # Store daily data
        daily_data[gauge_id] = df_period

        # Compute monthly means from daily data
        df_period = df_period.set_index("date")
        monthly = df_period["discharge_m3s"].resample("MS").mean()
        monthly_data[gauge_id] = monthly

        print(f"  Monthly records: {len(monthly)}")

    except Exception as e:
        print(f"Error loading data for {river_name} (gauge {gauge_id}): {e}")

# Compute summary statistics for each gauge over the 1985-1989 period
# Using the monthly streamflow values
summary_rows = []

for gauge_id, river_name in gauges.items():
    if gauge_id in monthly_data:
        monthly = monthly_data[gauge_id]
        # Drop NaN values for statistics
        valid = monthly.dropna()

        row = {
            "gauge_id": gauge_id,
            "river_name": river_name,
            "n_months": len(valid),
            "mean_m3s": valid.mean(),
            "std_m3s": valid.std(),
            "min_m3s": valid.min(),
            "max_m3s": valid.max(),
        }

        # Add metadata if available
        if metadata is not None:
            gauge_meta = metadata[metadata["gauge_id"] == int(gauge_id)]
            if len(gauge_meta) > 0:
                row["lat"] = gauge_meta["lat"].values[0]
                row["lon"] = gauge_meta["lon"].values[0]
                row["area_km2"] = gauge_meta["area_km2"].values[0]

        summary_rows.append(row)
        print(f"\n{river_name} ({gauge_id}):")
        print(f"  Mean:  {row['mean_m3s']:.2f} m3/s")
        print(f"  Std:   {row['std_m3s']:.2f} m3/s")
        print(f"  Min:   {row['min_m3s']:.2f} m3/s")
        print(f"  Max:   {row['max_m3s']:.2f} m3/s")
    else:
        print(f"\nNo data available for {river_name} ({gauge_id})")

# Create summary DataFrame and save
try:
    summary_df = pd.DataFrame(summary_rows)
    output_path = os.path.join(output_dir, "obs_streamflow_summary_1985_1989.csv")
    summary_df.to_csv(output_path, index=False, float_format="%.2f")
    print(f"\nSummary statistics saved to: {output_path}")
    print("\n" + summary_df.to_string(index=False))
except Exception as e:
    print(f"Error saving summary CSV: {e}")

# Also save the monthly time series for reference
try:
    monthly_df = pd.DataFrame(monthly_data)
    monthly_df.columns = [f"{gauges[gid]}_{gid}" for gid in monthly_df.columns]
    monthly_ts_path = os.path.join(output_dir, "obs_monthly_streamflow_1985_1989.csv")
    monthly_df.to_csv(monthly_ts_path, float_format="%.2f")
    print(f"Monthly time series saved to: {monthly_ts_path}")
except Exception as e:
    print(f"Error saving monthly time series CSV: {e}")

print("\nDone.")