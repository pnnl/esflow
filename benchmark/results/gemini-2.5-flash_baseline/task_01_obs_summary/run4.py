import os
import pandas as pd
import numpy as np

def analyze_grdc_streamflow(output_dir):
    """
    Loads GRDC gauge metadata, extracts monthly streamflow observations for
    specified gauges, computes summary statistics, and saves the results to CSV.
    """
    # Define input paths
    gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"
    streamflow_dir = "./data/sample/obs/streamflow/"

    # Define target gauges and their GRDC IDs
    target_gauges = {
        "Amazon": "3629000",
        "Missouri": "4121801",
        "Columbia": "4115200",
        "Danube": "6742900",
        "Mekong": "2969100",
        "Orange": "1159100"
    }

    # Define the analysis period
    start_date = '1985-01-01'
    end_date = '1989-12-31'

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return

    # Load gauge metadata
    try:
        gauge_metadata = pd.read_csv(gauge_metadata_path)
        print(f"Loaded gauge metadata from {gauge_metadata_path}")
    except FileNotFoundError:
        print(f"Error: Gauge metadata file not found at {gauge_metadata_path}")
        return
    except Exception as e:
        print(f"Error loading gauge metadata: {e}")
        return

    all_gauge_stats = []

    for river_name, gauge_id in target_gauges.items():
        print(f"Processing gauge: {river_name} (ID: {gauge_id})")
        streamflow_file = os.path.join(streamflow_dir, f"{gauge_id}.csv")

        try:
            # Load daily streamflow data
            df_daily = pd.read_csv(streamflow_file, parse_dates=['date'], index_col='date')
            df_daily = df_daily.loc[start_date:end_date] # Filter by period

            if df_daily.empty:
                print(f"Warning: No data found for {river_name} ({gauge_id}) in the period {start_date} to {end_date}. Skipping.")
                continue

            # Convert daily to monthly mean
            df_monthly = df_daily['discharge_m3s'].resample('MS').mean().to_frame()
            df_monthly.rename(columns={'discharge_m3s': 'monthly_mean_discharge_m3s'}, inplace=True)

            if df_monthly.empty:
                print(f"Warning: No monthly data could be computed for {river_name} ({gauge_id}). Skipping.")
                continue

            # Compute summary statistics
            mean_discharge = df_monthly['monthly_mean_discharge_m3s'].mean()
            std_discharge = df_monthly['monthly_mean_discharge_m3s'].std()
            min_discharge = df_monthly['monthly_mean_discharge_m3s'].min()
            max_discharge = df_monthly['monthly_mean_discharge_m3s'].max()

            # Store results
            all_gauge_stats.append({
                'gauge_id': gauge_id,
                'river_name': river_name,
                'mean_monthly_discharge_m3s': mean_discharge,
                'std_monthly_discharge_m3s': std_discharge,
                'min_monthly_discharge_m3s': min_discharge,
                'max_monthly_discharge_m3s': max_discharge
            })
            print(f"  - Statistics computed for {river_name}.")

        except FileNotFoundError:
            print(f"Error: Streamflow file not found for {river_name} at {streamflow_file}. Skipping.")
        except KeyError:
            print(f"Error: 'discharge_m3s' column not found in {streamflow_file}. Skipping.")
        except Exception as e:
            print(f"Error processing {river_name} ({gauge_id}): {e}. Skipping.")

    if not all_gauge_stats:
        print("No statistics were computed for any gauge. Exiting.")
        return

    # Combine all statistics into a single DataFrame
    summary_df = pd.DataFrame(all_gauge_stats)

    # Save the summary statistics to a CSV file
    output_csv_path = os.path.join(output_dir, "grdc_streamflow_summary_1985-1989.csv")
    try:
        summary_df.to_csv(output_csv_path, index=False)
        print(f"\nSummary statistics saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving summary statistics to CSV: {e}")

if __name__ == "__main__":
    output_directory = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_01_obs_summary/run4_output"
    analyze_grdc_streamflow(output_directory)