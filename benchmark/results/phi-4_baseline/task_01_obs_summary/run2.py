import os
import pandas as pd

# Define the input and output directories
input_dir = "./data/sample/obs"
output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/phi-4_baseline/task_01_obs_summary/run2_output"

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
    streamflow_path = os.path.join(input_dir, f"streamflow/{gauge_id}.csv")
    try:
        # Load the streamflow data
        df = pd.read_csv(streamflow_path)
        
        # Convert date column to datetime and filter for 1985-1989
        df['date'] = pd.to_datetime(df['date'])
        df_filtered = df[(df['date'].dt.year >= 1985) & (df['date'].dt.year <= 1989)]
        
        # Compute summary statistics
        mean_flow = df_filtered['discharge_m3s'].mean()
        std_flow = df_filtered['discharge_m3s'].std()
        min_flow = df_filtered['discharge_m3s'].min()
        max_flow = df_filtered['discharge_m3s'].max()
        
        # Store the results
        summary_stats[gauge_id] = {
            'mean': mean_flow,
            'std': std_flow,
            'min': min_flow,
            'max': max_flow
        }
    except Exception as e:
        print(f"Error processing gauge {gauge_id}: {e}")

# Convert the summary statistics to a DataFrame
summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')

# Add river names from metadata
summary_df['river_name'] = summary_df.index.map(gauge_metadata.set_index('gauge_id')['river_name'])

# Save the results as a CSV file
output_path = os.path.join(output_dir, "obs_summary.csv")
try:
    summary_df.to_csv(output_path)
except Exception as e:
    print(f"Error saving output: {e}")