import os
import pandas as pd
import numpy as np

# Output directory
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_01_obs_summary/run1_output"
os.makedirs(output_dir, exist_ok=True)

# Define gauges of interest
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
    print(metadata.head())
    print(f"Columns: {list(metadata.columns)}")
    print(f"Number of gauges in metadata: {len(metadata)}")
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    metadata = None

# Date range
start_date = "1985-01-01"
end_date = "1989-12-31"

# Store results
all_stats = []

for gauge_id, river_name in gauges.items():
    csv_path = f"./data/sample/obs/streamflow/{gauge_id}.csv"
    try:
        # Load daily observation data
        df = pd.read_csv(csv_path, parse_dates=["date"])
        print(f"\nLoaded {river_name} (gauge {gauge_id}): {len(df)} records")
        
        # Filter to 1985-1989
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df_period = df.loc[mask].copy()
        print(f"  Records in 1985-1989: {len(df_period)}")
        
        if len(df_period) == 0:
            print(f"  WARNING: No data in the specified period for {river_name}")
            continue
        
        # Drop NaN discharge values before computing monthly means
        df_period = df_period.dropna(subset=["discharge_m3s"])
        print(f"  Valid records after dropping NaN: {len(df_period)}")
        
        # Compute monthly means from daily data
        df_period["year"] = df_period["date"].dt.year
        df_period["month"] = df_period["date"].dt.month
        monthly = df_period.groupby(["year", "month"])["discharge_m3s"].mean().reset_index()
        monthly.rename(columns={"discharge_m3s": "monthly_mean_discharge_m3s"}, inplace=True)
        print(f"  Number of monthly values: {len(monthly)}")
        
        # Compute summary statistics over the monthly values
        discharge = monthly["monthly_mean_discharge_m3s"]
        stats = {
            "gauge_id": gauge_id,
            "river_name": river_name,
            "mean_m3s": discharge.mean(),
            "std_m3s": discharge.std(),
            "min_m3s": discharge.min(),
            "max_m3s": discharge.max(),
            "n_months": len(discharge),
        }
        
        # Add metadata if available
        if metadata is not None:
            # Try matching gauge_id (could be int or string)
            meta_row = metadata[metadata["gauge_id"].astype(str) == str(gauge_id)]
            if len(meta_row) > 0:
                stats["lat"] = meta_row["lat"].values[0]
                stats["lon"] = meta_row["lon"].values[0]
                stats["area_km2"] = meta_row["area_km2"].values[0]
        
        all_stats.append(stats)
        print(f"  Mean: {stats['mean_m3s']:.2f} m3/s, Std: {stats['std_m3s']:.2f} m3/s, "
              f"Min: {stats['min_m3s']:.2f} m3/s, Max: {stats['max_m3s']:.2f} m3/s")
        
    except Exception as e:
        print(f"Error processing gauge {gauge_id} ({river_name}): {e}")

# Save summary statistics to CSV
try:
    stats_df = pd.DataFrame(all_stats)
    output_path = os.path.join(output_dir, "obs_streamflow_summary_1985_1989.csv")
    stats_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"\nSummary statistics saved to: {output_path}")
    print("\n--- Summary Table ---")
    print(stats_df.to_string(index=False))
except Exception as e:
    print(f"Error saving summary CSV: {e}")