import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
GAUGE_METADATA_PATH = "./data/sample/obs/gauge_metadata.csv"
STREAMFLOW_DIR = "./data/sample/obs/streamflow"
OUTPUT_DIR = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/claude-haiku-4-5-20251001_baseline/task_01_obs_summary/run2_output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define gauges of interest
GAUGES = {
    "Amazon": 3629000,
    "Missouri": 4121801,
    "Columbia": 4115200,
    "Danube": 6742900,
    "Mekong": 2969100,
    "Orange": 1159100
}

# Time period
START_DATE = "1985-01-01"
END_DATE = "1989-12-31"

try:
    # Load gauge metadata
    print("Loading gauge metadata...")
    metadata = pd.read_csv(GAUGE_METADATA_PATH)
    print(f"Loaded metadata for {len(metadata)} gauges")
    
    # Initialize results storage
    summary_stats = []
    
    # Process each gauge
    for gauge_name, gauge_id in GAUGES.items():
        print(f"\nProcessing {gauge_name} (ID: {gauge_id})...")
        
        # Construct path to gauge CSV
        gauge_csv_path = os.path.join(STREAMFLOW_DIR, f"{gauge_id}.csv")
        
        if not os.path.exists(gauge_csv_path):
            print(f"  WARNING: File not found: {gauge_csv_path}")
            continue
        
        try:
            # Load streamflow data
            df = pd.read_csv(gauge_csv_path)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to time period of interest
            df_filtered = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)].copy()
            
            if len(df_filtered) == 0:
                print(f"  WARNING: No data found in time period {START_DATE} to {END_DATE}")
                continue
            
            # Extract discharge column (handle potential variations in column naming)
            discharge_col = None
            for col in df_filtered.columns:
                if 'discharge' in col.lower() or 'q' in col.lower():
                    discharge_col = col
                    break
            
            if discharge_col is None:
                print(f"  WARNING: Could not find discharge column in {gauge_csv_path}")
                print(f"  Available columns: {df_filtered.columns.tolist()}")
                continue
            
            # Remove NaN values
            discharge = df_filtered[discharge_col].dropna()
            
            if len(discharge) == 0:
                print(f"  WARNING: No valid discharge data after removing NaNs")
                continue
            
            # Compute summary statistics
            stats = {
                'gauge_name': gauge_name,
                'gauge_id': gauge_id,
                'n_observations': len(discharge),
                'mean_m3s': discharge.mean(),
                'std_m3s': discharge.std(),
                'min_m3s': discharge.min(),
                'max_m3s': discharge.max(),
                'median_m3s': discharge.median(),
                'q25_m3s': discharge.quantile(0.25),
                'q75_m3s': discharge.quantile(0.75)
            }
            
            summary_stats.append(stats)
            
            print(f"  Mean discharge: {stats['mean_m3s']:.2f} m³/s")
            print(f"  Std deviation: {stats['std_m3s']:.2f} m³/s")
            print(f"  Min: {stats['min_m3s']:.2f} m³/s")
            print(f"  Max: {stats['max_m3s']:.2f} m³/s")
            print(f"  N observations: {stats['n_observations']}")
            
        except Exception as e:
            print(f"  ERROR processing {gauge_name}: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(summary_stats)
    
    # Save results to CSV
    output_csv = os.path.join(OUTPUT_DIR, "obs_streamflow_summary_1985_1989.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Summary statistics saved to: {output_csv}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR OBSERVED STREAMFLOW (1985-1989)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
except Exception as e:
    print(f"FATAL ERROR: {str(e)}")
    import traceback
    traceback.print_exc()