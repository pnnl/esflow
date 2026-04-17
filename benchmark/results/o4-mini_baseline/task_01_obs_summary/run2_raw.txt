import os
import pandas as pd

def main():
    # File paths and parameters
    gauge_meta_file = "./data/sample/obs/gauge_metadata.csv"
    streamflow_dir = "./data/sample/obs/streamflow"
    gauge_ids = ["3629000", "4121801", "4115200", "6742900", "2969100", "1159100"]
    start_date = "1985-01-01"
    end_date = "1989-12-31"
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_01_obs_summary/run2_output"
    output_file = os.path.join(output_dir, "obs_streamflow_summary.csv")

    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")

    # Load gauge metadata
    try:
        meta = pd.read_csv(gauge_meta_file, dtype={"gauge_id": str})
    except Exception as e:
        print(f"Error reading gauge metadata file {gauge_meta_file}: {e}")
        return

    # Filter metadata for selected gauges
    meta_sel = meta[meta["gauge_id"].isin(gauge_ids)].set_index("gauge_id")

    # Prepare summary list
    summary_records = []

    # Loop over gauges
    for gid in gauge_ids:
        csv_file = os.path.join(streamflow_dir, f"{gid}.csv")
        try:
            df = pd.read_csv(csv_file, parse_dates=["date"])
        except Exception as e:
            print(f"Error reading streamflow file for gauge {gid}: {e}")
            continue

        # Ensure correct columns
        if "date" not in df.columns or "discharge_m3s" not in df.columns:
            print(f"Missing required columns in {csv_file}")
            continue

        # Filter date range
        df = df.set_index("date").sort_index()
        df = df[start_date:end_date]

        # Check if data is empty
        if df.empty:
            print(f"No data for gauge {gid} in period {start_date} to {end_date}")
            continue

        # Resample to monthly mean streamflow
        monthly = df["discharge_m3s"].resample("MS").mean()

        # Compute summary statistics
        rec = {
            "gauge_id": gid,
            "river_name": meta_sel.loc[gid, "river_name"] if gid in meta_sel.index else "",
            "mean_m3s": monthly.mean(),
            "std_m3s": monthly.std(),
            "min_m3s": monthly.min(),
            "max_m3s": monthly.max(),
        }
        summary_records.append(rec)

    # Create summary DataFrame
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        # Save to CSV
        try:
            summary_df.to_csv(output_file, index=False)
            print(f"Saved summary to {output_file}")
        except Exception as e:
            print(f"Error saving summary CSV {output_file}: {e}")
    else:
        print("No summary records to save.")

if __name__ == "__main__":
    main()