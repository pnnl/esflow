import os
import pandas as pd
import numpy as np

output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-opus-4-6_baseline/task_01_obs_summary/run3_output"
os.makedirs(output_dir, exist_ok=True)

# Define the six gauges of interest
gauge_ids = {
    3629000: "Amazon",
    4121801: "Missouri",
    4115200: "Columbia",
    6742900: "Danube",
    2969100: "Mekong",
    1159100: "Orange",
}

# Load gauge metadata
try:
    metadata = pd.read_csv("./data/sample/obs/gauge_metadata.csv")
    print("Gauge metadata loaded successfully.")
    print(metadata.head())
    print(f"Columns: {list(metadata.columns)}")
except Exception as e:
    print(f"Error loading gauge metadata: {e}")
    metadata = None

# Date range for analysis
start_date = "1985-01-01"
end_date = "1989-12-31"

# Collect monthly streamflow for each gauge
results = []

for gauge_id, river_name in gauge_ids.items():
    csv_path = f"./data/sample/obs/streamflow/{gauge_id}.csv"
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        print(f"\nLoaded data for {river_name} (gauge {gauge_id}): {len(df)} records")
        
        # Filter to the 1985-1989 period
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df_period = df.loc[mask].copy()
        print(f"  Records in 1985-1989: {len(df_period)}")
        
        if len(df_period) == 0:
            print(f"  WARNING: No data found for {river_name} in 1985-1989.")
            continue
        
        # Resample daily data to monthly means
        df_period = df_period.set_index("date")
        monthly = df_period["discharge_m3s"].resample("MS").mean()
        monthly = monthly.dropna()
        print(f"  Monthly values: {len(monthly)}")
        
        # Compute summary statistics
        mean_val = monthly.mean()
        std_val = monthly.std()
        min_val = monthly.min()
        max_val = monthly.max()
        
        # Get metadata info if available
        lat, lon, area_km2 = np.nan, np.nan, np.nan
        if metadata is not None:
            meta_row = metadata[metadata["gauge_id"] == gauge_id]
            if len(meta_row) > 0:
                lat = meta_row["lat"].values[0]
                lon = meta_row["lon"].values[0]
                area_km2 = meta_row["area_km2"].values[0]
        
        results.append({
            "gauge_id": gauge_id,
            "river_name": river_name,
            "lat": lat,
            "lon": lon,
            "area_km2": area_km2,
            "n_months": len(monthly),
            "mean_discharge_m3s": mean_val,
            "std_discharge_m3s": std_val,
            "min_discharge_m3s": min_val,
            "max_discharge_m3s": max_val,
        })
        
        print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f} m3/s")
        
    except Exception as e:
        print(f"Error processing gauge {gauge_id} ({river_name}): {e}")

# Create summary DataFrame and save
try:
    summary_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, "obs_streamflow_summary_1985_1989.csv")
    summary_df.to_csv(output_csv, index=False, float_format="%.4f")
    print(f"\nSummary statistics saved to: {output_csv}")
    print("\n--- Summary Table ---")
    print(summary_df.to_string(index=False))
except Exception as e:
    print(f"Error saving summary CSV: {e}")