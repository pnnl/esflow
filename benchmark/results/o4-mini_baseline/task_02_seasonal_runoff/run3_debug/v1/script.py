import os
import glob
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def main():
    # Configuration
    data_dir = "./data/sample/e3sm/lnd"
    case_name = "sample.v3.LR.historical"
    start_year = 1985
    end_year = 1989
    outdir = "/Users/zhou014/Local_Drive/Git_repo/esmflow/benchmark/results/o4-mini_baseline/task_02_seasonal_runoff/run3_debug/v1/output"
    os.makedirs(outdir, exist_ok=True)

    # Collect files
    try:
        pattern = os.path.join(data_dir, f"{case_name}.elm.h0.*.nc")
        all_files = glob.glob(pattern)
        files = []
        for fp in all_files:
            m = re.search(r"\.(\d{4})-(\d{2})\.nc$", fp)
            if m:
                year = int(m.group(1))
                if start_year <= year <= end_year:
                    files.append(fp)
        files = sorted(files)
        if not files:
            print(f"No files found for years {start_year}-{end_year}")
            return
        print(f"Found {len(files)} files for period {start_year}-{end_year}")
    except Exception as e:
        print(f"Error collecting files: {e}")
        return

    # Open dataset
    try:
        ds = xr.open_mfdataset(files, combine='by_coords', parallel=False)
    except Exception as e:
        print(f"Error opening datasets: {e}")
        return

    # Extract QRUNOFF
    varname = "QRUNOFF"
    if varname not in ds:
        print(f"Variable {varname} not found in dataset")
        return
    qrunoff = ds[varname]

    # Compute climatological mean
    try:
        climatology = qrunoff.mean(dim='time', skipna=True)
    except Exception as e:
        print(f"Error computing climatology: {e}")
        return

    # Compute area weights
    weights = None
    if 'AREA' in ds:
        weights = ds['AREA']
    elif 'area' in ds:
        weights = ds['area']
    else:
        if 'lat' in ds:
            try:
                lat = ds['lat']
                coslat = np.cos(np.deg2rad(lat))
                if coslat.ndim == 1 and 'lon' in ds:
                    coslat = coslat.broadcast_like(climatology)
                weights = xr.DataArray(coslat, dims=climatology.dims, coords=climatology.coords)
                print("Using cosine latitude weights")
            except Exception as e:
                print(f"Error computing cos(lat) weights: {e}")
        else:
            print("No area or lat coordinate for weighting; global mean will be simple average")

    # Compute global mean runoff
    try:
        if weights is not None and set(['lat','lon']).issubset(climatology.dims):
            gwmean_da = (climatology * weights).sum(dim=['lat','lon'], skipna=True) / weights.where(~np.isnan(climatology)).sum(dim=['lat','lon'], skipna=True)
        else:
            gwmean_da = climatology.mean(dim=[d for d in climatology.dims], skipna=True)
    except Exception as e:
        print(f"Error computing global mean: {e}")
        return

    # Convert global mean to scalar
    try:
        arr = gwmean_da.values
        if np.size(arr) == 1:
            global_mean = float(arr)
        else:
            global_mean = float(np.nanmean(arr))
    except Exception:
        try:
            global_mean = float(gwmean_da.item())
        except Exception:
            global_mean = float(np.nanmean(np.array(gwmean_da)))
    print(f"Global mean QRUNOFF climatology (1985-1989): {global_mean:.6e}")

    # Save climatology to NetCDF
    out_nc = os.path.join(outdir, "qrunoff_climatology_1985-1989.nc")
    try:
        climatology.to_dataset(name=varname).to_netcdf(out_nc)
        print(f"Saved climatology to {out_nc}")
    except Exception as e:
        print(f"Error saving NetCDF: {e}")

    # Save global mean to text
    out_txt = os.path.join(outdir, "global_mean_qrunoff_1985-1989.txt")
    try:
        with open(out_txt, "w") as f:
            f.write(f"Global mean QRUNOFF climatology (1985-1989): {global_mean:.6e} mm/s\n")
        print(f"Saved global mean to {out_txt}")
    except Exception as e:
        print(f"Error saving global mean text: {e}")

    # Plot the climatology field
    out_png = os.path.join(outdir, "qrunoff_climatology_map.png")
    try:
        plt.figure(figsize=(10,5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        pcm = climatology.plot.pcolormesh(
            ax=ax, transform=ccrs.PlateCarree(), cmap='viridis',
            cbar_kwargs={'label':'QRUNOFF (mm/s)'}, add_colorbar=True)
        title_txt = f"QRUNOFF Climatology (1985-1989)\nGlobal mean = {global_mean:.2e} mm/s"
        ax.set_title(title_txt)
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {out_png}")
    except Exception as e:
        print(f"Error plotting climatology: {e}")

if __name__ == "__main__":
    main()