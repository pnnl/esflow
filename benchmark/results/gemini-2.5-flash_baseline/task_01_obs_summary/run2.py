import os
import pandas as pd
import numpy as np

def analyze_grdc_streamflow(output_dir):
    """
    Loads GRDC gauge metadata, extracts monthly streamflow observations for
    specified gauges, computes summary statistics, and saves the results to CSV.
    """
    # Define input and output paths
    data_dir = "./data/sample/obs/"
    gauge_metadata_path = os.path.join(data_dir, "gauge_metadata.csv")
    streamflow_dir = os.path.join(data_dir, "streamflow/")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "grdc_streamflow_summary.csv")

    # Define target gauges and period
    target_gauge_ids = {
        3629000: "Amazon",
        4121801: "Missouri",
        4115200: "Columbia",
        6742900: "Danube",
        2969100: "Mekong",
        1159100: "Orange"
    }
    start_date = '1985-01-01'
    end_date = '1989-12-31'

    print(f"Starting GRDC streamflow analysis for {len(target_gauge_ids)} gauges...")

    try:
        # Load gauge metadata
        gauge_metadata = pd.read_csv(gauge_metadata_path, index_col='gauge_id')
        print(f"Loaded gauge metadata from {gauge_metadata_path}")
    except Exception as e:
        print(f"Error loading gauge metadata from {gauge_metadata_path}: {e}")
        return

    summary_data = []

    for gauge_id, river_name in target_gauge_ids.items():
        print(f"Processing gauge {gauge_id} ({river_name})...")
        streamflow_file = os.path.join(streamflow_dir, f"{gauge_id}.csv")

        try:
            # Load daily streamflow data
            df_daily = pd.read_csv(streamflow_file, parse_dates=['date'], index_col='date')
            df_daily = df_daily.loc[start_date:end_date]

            if df_daily.empty:
                print(f"  No data found for {gauge_id} in the period {start_date} to {end_date}. Skipping.")
                continue

            # Convert daily to monthly mean
            df_monthly = df_daily['discharge_m3s'].resample('MS').mean().to_frame()
            df_monthly.rename(columns={'discharge_m3s': 'monthly_discharge_m3s'}, inplace=True)

            # Compute summary statistics
            mean_q = df_monthly['monthly_discharge_m3s'].mean()
            std_q = df_monthly['monthly_discharge_m3s'].std()
            min_q = df_monthly['monthly_discharge_m3s'].min()
            max_q = df_monthly['monthly_discharge_m3s'].max()

            # Get metadata for the current gauge
            gauge_info = gauge_metadata.loc[gauge_id]

            summary_data.append({
                'gauge_id': gauge_id,
                'river_name': river_name,
                'lat': gauge_info['lat'],
                'lon': gauge_info['lon'],
                'area_km2': gauge_info['area_km2'],
                'mean_discharge_m3s': mean_q,
                'std_discharge_m3s': std_q,
                'min_discharge_m3s': min_q,
                'max_discharge_m3s': max_q
            })
            print(f"  Successfully processed {gauge_id}.")

        except FileNotFoundError:
            print(f"  Streamflow file not found for gauge {gauge_id}: {streamflow_file}. Skipping.")
        except Exception as e:
            print(f"  Error processing streamflow for gauge {gauge_id} from {streamflow_file}: {e}. Skipping.")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        try:
            summary_df.to_csv(output_csv_path, index=False)
            print(f"Summary statistics saved to {output_csv_path}")
        except Exception as e:
            print(f"Error saving summary statistics to {output_csv_path}: {e}")
    else:
        print("No data processed for any of the target gauges. No summary CSV created.")

    print("GRDC streamflow analysis complete.")

if __name__ == "__main__":
    output_directory = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_01_obs_summary/run2_output"
    analyze_grdc_streamflow(output_directory)