import os
import pandas as pd
import numpy as np
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def load_gauge_metadata(metadata_path):
    try:
        df_meta = pd.read_csv(metadata_path, dtype={'gauge_id': str})
        df_meta.set_index('gauge_id', inplace=True)
        logging.info(f"Loaded gauge metadata with {len(df_meta)} entries")
        return df_meta
    except Exception as e:
        logging.error(f"Error loading gauge metadata: {e}")
        raise

def process_gauge_streamflow(gauge_id, streamflow_dir, start_date, end_date):
    """
    Load daily streamflow for a gauge, filter by date, compute monthly mean.
    Returns a pandas Series of monthly mean discharge.
    """
    fname = os.path.join(streamflow_dir, f"{gauge_id}.csv")
    try:
        df = pd.read_csv(fname, parse_dates=['date'])
    except Exception as e:
        logging.error(f"Failed to read streamflow file for gauge {gauge_id}: {e}")
        return None

    if 'discharge_m3s' not in df.columns:
        logging.error(f"'discharge_m3s' column not found in {fname}")
        return None

    df = df.set_index('date').sort_index()
    # Filter dates
    df = df.loc[start_date:end_date]
    if df.empty:
        logging.warning(f"No data for gauge {gauge_id} in period {start_date} to {end_date}")
        return None
    # Resample to monthly mean
    monthly = df['discharge_m3s'].resample('M').mean()
    return monthly

def compute_summary_statistics(series):
    """
    Given a pandas Series, compute mean, std, min, max.
    Returns a dict.
    """
    return {
        'mean_m3s': series.mean(),
        'std_m3s': series.std(),
        'min_m3s': series.min(),
        'max_m3s': series.max()
    }

def main():
    setup_logging()

    # Parameters
    metadata_path = "./data/sample/obs/gauge_metadata.csv"
    streamflow_dir = "./data/sample/obs/streamflow"
    gauge_ids = ['3629000', '4121801', '4115200', '6742900', '2969100', '1159100']
    start_date = '1985-01-01'
    end_date = '1989-12-31'
    output_dir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_01_obs_summary/run4_output"
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    df_meta = load_gauge_metadata(metadata_path)

    # Prepare list for summary
    records = []

    for gid in gauge_ids:
        logging.info(f"Processing gauge {gid}")
        monthly_series = process_gauge_streamflow(gid, streamflow_dir, start_date, end_date)
        if monthly_series is None:
            logging.warning(f"Skipping gauge {gid} due to missing data")
            continue
        stats = compute_summary_statistics(monthly_series)
        meta = df_meta.loc[gid] if gid in df_meta.index else {}
        record = {
            'gauge_id': gid,
            'river_name': meta.get('river_name', ''),
            'lat': meta.get('lat', np.nan),
            'lon': meta.get('lon', np.nan),
            'area_km2': meta.get('area_km2', np.nan),
            'mean_m3s': stats['mean_m3s'],
            'std_m3s': stats['std_m3s'],
            'min_m3s': stats['min_m3s'],
            'max_m3s': stats['max_m3s']
        }
        records.append(record)

    # Create DataFrame
    df_summary = pd.DataFrame.from_records(records,
                                           columns=['gauge_id','river_name','lat','lon','area_km2',
                                                    'mean_m3s','std_m3s','min_m3s','max_m3s'])
    summary_csv = os.path.join(output_dir, "obs_streamflow_summary_1985_1989.csv")
    try:
        df_summary.to_csv(summary_csv, index=False)
        logging.info(f"Saved summary statistics to {summary_csv}")
    except Exception as e:
        logging.error(f"Failed to save summary CSV: {e}")

if __name__ == "__main__":
    main()