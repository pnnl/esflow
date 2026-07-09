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
    streamflow_data_dir = os.path.join(data_dir, "streamflow/")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "grdc_streamflow_summary.csv")

    # Define target gauges and period
    target_gauge_ids = {
        "Amazon": 3629000,
        "Missouri": 4121801,
        "Columbia": 4115200,
        "Danube": 6742900,
        "Mekong": 2969100,
        "Orange": 1159100
    }
    start_date = '1985-01-01'
    end_date = '1989-12-31'

    print(f"Starting GRDC streamflow analysis for {len(target_gauge_ids)} gauges...")
    print(f"Period: {start_date} to {end_date}")

    try:
        # Load gauge metadata
        gauge_metadata = pd.read_csv(gauge_metadata_path, index_col='gauge_id')
        print(f"Loaded gauge metadata from: {gauge_metadata_path}")
    except Exception as e:
        print(f"Error loading gauge metadata from {gauge_metadata_path}: {e}")
        return

    all_gauge_summary = []

    for gauge_name, gauge_id in target_gauge_ids.items():
        print(f"Processing gauge: {gauge_name} (ID: {gauge_id})...")
        gauge_file_path = os.path.join(streamflow_data_dir, f"{gauge_id}.csv")

        try:
            # Load daily streamflow data for the current gauge
            df_gauge = pd.read_csv(gauge_file_path, parse_dates=['date'], index_col='date')
            
            # Filter for the specified period
            df_gauge_period = df_gauge.loc[start_date:end_date]

            # Convert daily to monthly mean
            # Resample to 'MS' (month start) and calculate the mean
            monthly_discharge = df_gauge_period['discharge_m3s'].resample('MS').mean()

            # Compute summary statistics
            mean_q = monthly_discharge.mean()
            std_q = monthly_discharge.std()
            min_q = monthly_discharge.min()
            max_q = monthly_discharge.max()

            # Get metadata for the current gauge
            if gauge_id in gauge_metadata.index:
                lat = gauge_metadata.loc[gauge_id, 'lat']
                lon = gauge_metadata.loc[gauge_id, 'lon']
                area_km2 = gauge_metadata.loc[gauge_id, 'area_km2']
                river_name = gauge_metadata.loc[gauge_id, 'river_name']
            else:
                lat, lon, area_km2, river_name = np.nan, np.nan, np.nan, np.nan
                print(f"Warning: Metadata not found for gauge ID {gauge_id}.")

            all_gauge_summary.append({
                'gauge_id': gauge_id,
                'gauge_name': gauge_name,
                'river_name': river_name,
                'lat': lat,
                'lon': lon,
                'area_km2': area_km2,
                'mean_discharge_m3s': mean_q,
                'std_discharge_m3s': std_q,
                'min_discharge_m3s': min_q,
                'max_discharge_m3s': max_q
            })
            print(f"  Successfully processed {gauge_name}.")

        except FileNotFoundError:
            print(f"Error: Streamflow file not found for gauge ID {gauge_id} at {gauge_file_path}")
        except KeyError:
            print(f"Error: 'discharge_m3s' column not found in {gauge_file_path} or date parsing issue.")
        except Exception as e:
            print(f"An unexpected error occurred while processing gauge {gauge_id} ({gauge_name}): {e}")

    if all_gauge_summary:
        summary_df = pd.DataFrame(all_gauge_summary)
        try:
            summary_df.to_csv(output_csv_path, index=False)
            print(f"\nSummary statistics saved to: {output_csv_path}")
        except Exception as e:
            print(f"Error saving summary statistics to {output_csv_path}: {e}")
    else:
        print("\nNo gauge data was successfully processed to generate a summary.")

if __name__ == "__main__":
    # Define the output directory as specified in the problem description
    output_directory = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_01_obs_summary/run1_output"
    analyze_grdc_streamflow(output_directory)