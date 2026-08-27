FROM python:3.10-slim

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --create-home --uid 1000 benchmark

# Pre-fetch the Natural Earth shapefiles v2's own plotters use (110m land,
# ocean, coastline, borders, rivers, lakes) so baseline scripts that render
# a coastline map don't need network access at container-run time. This
# covers the 5 tasks that keep --network=none; task_03_et_benchmark and
# task_07_integrated_diagnostic get full outbound network instead (see
# NETWORK_REQUIRED_TASKS in benchmark/common.py) and can fetch any
# resolution/feature a generated script happens to request.
ENV CARTOPY_DATA_DIR=/opt/cartopy-data
RUN mkdir -p "$CARTOPY_DATA_DIR" && chmod 755 "$CARTOPY_DATA_DIR" \
    && python -c "\
import cartopy; cartopy.config['data_dir'] = '$CARTOPY_DATA_DIR'; \
import cartopy.io.shapereader as shpreader; \
[shpreader.natural_earth(resolution='110m', category='physical', name=n) for n in ('coastline', 'land', 'ocean', 'rivers_lake_centerlines', 'lakes')]; \
[shpreader.natural_earth(resolution='110m', category='cultural', name=n) for n in ('admin_0_boundary_lines_land',)]" \
    && chmod -R a+rX "$CARTOPY_DATA_DIR"

USER benchmark
