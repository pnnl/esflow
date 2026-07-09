import os
import pandas as pd

# Define the input and output directories
input_dir = "./data/sample/obs"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_01_obs_summary/run3_output"

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Load gauge metadata
gauge_metadata_path = os.path.join(input_dir, "gauge_metadata.csv")
try:
    gauge_metadata = pd.read_csv(gauge_metadata_path)
except Exception as e:
    print(f"Error reading gauge metadata: {e}")
    raise

# List of gauges to process
gauges_of_interest = ["3629000", "4121801", "4115200", "6742900", "2969100", "1159100"]

# Initialize a dictionary to store summary statistics for each gauge
summary_stats = {}

# Process each gauge
for gauge_id in gauges_of_interest:
    try:
        # Load the streamflow data for the current gauge
        streamflow_path = os.path.join(input_dir, f"streamflow/{gauge_id}.csv")
        streamflow_data = pd.read_csv(streamflow_path)
        
        # Convert date column to datetime and filter data from 1985 to 1989
        streamflow_data['date'] = pd.to_datetime(streamflow_data['date'])
        filtered_data = streamflow_data[(streamflow_data['date'].dt.year >= 1985) & (streamflow_data['date'].dt.year <= 1989)]
        
        # Compute summary statistics
        mean_discharge = filtered_data['discharge_m3s'].mean()
        std_discharge = filtered_data['discharge_m3s'].std()
        min_discharge = filtered_data['discharge_m3s'].min()
        max_discharge = filtered_data['discharge_m3s'].max()
        
        # Store the results in the summary_stats dictionary
        summary_stats[gauge_id] = {
            'mean': mean_discharge,
            'std': std_discharge,
            'min': min_discharge,
            'max': max_discharge
        }
    except Exception as e:
        print(f"Error processing gauge {gauge_id}: {e}")

# Convert the summary statistics dictionary to a DataFrame
summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')

# Save the results to a CSV file in the output directory
output_path = os.path.join(output_dir, "obs_summary.csv")
try:
    summary_df.to_csv(output_path)
except Exception as e:
    print(f"Error saving summary statistics: {e}")