import os
import pandas as pd
import numpy as np

def analyze_streamflow_observations(output_dir):
    """
    Loads GRDC gauge metadata, extracts monthly streamflow observations for
    specific gauges, computes summary statistics, and saves the results to CSV.

    Args:
        output_dir (str): The directory to save output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    gauge_metadata_path = "./data/sample/obs/gauge_metadata.csv"
    streamflow_data_dir = "./data/sample/obs/streamflow/"
    
    # Define target gauges and their GRDC IDs
    target_gauges = {
        "Amazon": "3629000",
        "Missouri": "4121801",
        "Columbia": "4115200",
        "Danube": "6742900",
        "Mekong": "2969100",
        "Orange": "1159100"
    }

    start_date = '1985-01-01'
    end_date = '1989-12-31'

    all_gauge_stats = []

    try:
        # Load gauge metadata
        gauge_metadata = pd.read_csv(gauge_metadata_path)
        print(f"Loaded gauge metadata from {gauge_metadata_path}")
    except Exception as e:
        print(f"Error loading gauge metadata from {gauge_metadata_path}: {e}")
        return

    for gauge_name, gauge_id in target_gauges.items():
        gauge_file_path = os.path.join(streamflow_data_dir, f"{gauge_id}.csv")
        try:
            # Load daily streamflow data
            df_daily = pd.read_csv(gauge_file_path, parse_dates=['date'], index_col='date')
            
            # Filter by date range
            df_filtered = df_daily.loc[start_date:end_date]
            
            # Convert to monthly mean
            df_monthly = df_filtered['discharge_m3s'].resample('MS').mean()
            
            if not df_monthly.empty:
                # Compute summary statistics
                mean_q = df_monthly.mean()
                std_q = df_monthly.std()
                min_q = df_monthly.min()
                max_q = df_monthly.max()

                all_gauge_stats.append({
                    'gauge_name': gauge_name,
                    'gauge_id': gauge_id,
                    'mean_discharge_m3s': mean_q,
                    'std_discharge_m3s': std_q,
                    'min_discharge_m3s': min_q,
                    'max_discharge_m3s': max_q
                })
                print(f"Processed gauge: {gauge_name} ({gauge_id})")
            else:
                print(f"No data found for {gauge_name} ({gauge_id}) in the period {start_date} to {end_date}.")

        except FileNotFoundError:
            print(f"Streamflow file not found for gauge {gauge_name} ({gauge_id}): {gauge_file_path}")
        except Exception as e:
            print(f"Error processing streamflow for gauge {gauge_name} ({gauge_id}): {e}")

    if all_gauge_stats:
        summary_df = pd.DataFrame(all_gauge_stats)
        output_csv_path = os.path.join(output_dir, "monthly_streamflow_summary_1985-1989.csv")
        try:
            summary_df.to_csv(output_csv_path, index=False)
            print(f"Summary statistics saved to {output_csv_path}")
        except Exception as e:
            print(f"Error saving summary statistics to {output_csv_path}: {e}")
    else:
        print("No summary statistics to save.")

if __name__ == "__main__":
    output_directory = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/gemini-2.5-flash_baseline/task_01_obs_summary/run3_output"
    analyze_streamflow_observations(output_directory)