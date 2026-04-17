# Failure Causes — Diagnostic Notes

Running log of root causes identified during manual review of undetermined
benchmark runs. Each entry links a specific run (or class of runs) to the
underlying mistake, so we can cite concrete examples in the paper and
catalog design lessons.

---

## o4-mini: `years: [1985, 1989]` interpreted as list-of-2, not range

**Affected runs:**
- `o4-mini` / protocol / `task_02_seasonal_runoff` / run4 → silent failure (idx 56)
- `o4-mini` / protocol / `task_04_streamflow_fdc` / run2 → silent failure (idx 57)
- `o4-mini` / protocol / `task_04_streamflow_fdc` / run4 → silent failure (idx 58)
- `o4-mini` / protocol / `task_06_water_balance` / runs 1-4 → silent failure (idx 72-75)
- `o4-mini` / protocol / `task_07_integrated_diagnostic` / run1 → silent failure (idx 76)

**Symptoms:**
- T2: 1.5% bias in global mean runoff (6.748e-6 vs 6.646e-6)
- T4: volume_bias shifts from 1.049 to 1.379; FDC shape distorted
  because only ~730 days of discharge used instead of ~1826

**Root cause:** o4-mini wrote `years: [1985, 1989]` in its YAML, apparently
intending it as a range "1985 through 1989". The `extract_gridded_field`
tool interprets `years` as a literal list of years to load, so it loaded
only the 1985 and 1989 files (24 months total) instead of all five years
(60 months). 1985 and 1989 happen to sit slightly above the 5-year mean,
producing the small but real ~1.5% drift.

**Reference (claude-opus-4-6 protocol run2):** explicitly enumerated all
five years: `years: [1985, 1986, 1987, 1988, 1989]`.

**Why this matters for the paper:**
- Classic parameter-level silent failure: workflow runs to completion,
  produces a plausible figure, but is quantitatively wrong.
- The error is *visible in the YAML* — a human reviewer or a static
  schema check could catch it. This is exactly the failure mode the
  module-grounded design is supposed to expose.
- Suggests a catalog design lesson: parameters with ambiguous
  list-vs-range semantics (`years`, `months`, `levels`, ...) should be
  split into explicit `year_start`/`year_end` vs `years_list`, or the
  schema should require a discriminator.

**Evidence:**
- YAML: `benchmark/results/o4-mini_protocol/task_02_seasonal_runoff/run4.yaml`
- Log line: `extract_runoff_field: ... 'years': [1985, 1989] ... Files found: 24 ... Time steps: 24`

**Watch list:** check whether o4-mini repeats this `[start, end]` pattern
in other tasks that take a `years` argument (T3, T6, T7).
**Confirmed:** pattern repeats across T2, T4, T6, and T7 — systematic o4-mini behavior.

---

## claude-opus-4-6: subdirectory output paths crash (T7 run1)

**Affected runs:**
- `claude-opus-4-6` / protocol / `task_07_integrated_diagnostic` / run1 → crash

**Symptom:** No PNG or summary CSV produced. All 23 workflow steps
failed in cascade.

**Root cause:** Opus run1 organized output paths into subdirectories
(`fields/`, `basins/`, `metadata/`, `timeseries/`, `metrics/`,
`summary/`, `plots/`) within the output directory, e.g.:
```yaml
field_file: ${settings.output_dir}/fields/model_precip.nc
```
The workflow engine does not auto-create intermediate subdirectories.
The first tool produced a file in the flat output dir, but the rename
into `fields/` failed (`No such file or directory`). Every downstream
step that referenced the missing output also failed → complete cascade.

**The YAML was scientifically correct** — all 5 years enumerated, proper
variables, correct tool sequence. Runs 2-4 used flat output paths and
all succeeded.

**Why this matters for the paper:**
- Illustrates the **step-multiplication risk** in multi-step workflows.
  If each step has a per-step success probability *p*, a workflow with
  *n* steps has end-to-end success probability *p^n*. For example, if
  *p* = 0.99 (99% per step), a 23-step workflow like T7 drops to
  0.99^23 ≈ 79%. Even small per-step failure rates compound quickly.
- The crash was caused by a formatting detail (subdirectory paths), not
  a scientific mistake — showing that workflow composition introduces
  failure modes beyond the science itself.
- Design lesson: the engine (or schema) should either auto-create
  output directories or reject subdirectory paths at validation time.

---

## Baseline T1: daily-to-monthly aggregation before summary stats

**Affected runs:**
- 19 of 24 baseline T1 runs → silent failure (idx 1-17, 19-20)
- Models: claude-opus-4-6 (4/4), gpt-5 (4/4), gemini-2.5-flash (4/4),
  o4-mini (4/4), claude-haiku-4-5 (3/4 — run2 correct)

**Passes:** phi-4 (4/4) and claude-haiku-4-5 run2 → computed on raw
daily data, matching reference.

**Symptom:** Summary statistics computed on n=60 monthly means instead
of n=1826 daily values. Most gauges differ by <0.5%, but Orange River
(high daily variability, std ≈ 2.4× mean) is 1.47% off — exceeding 1%
tolerance.

**Root cause:** The task prompt asks for "monthly summary statistics"
and the system prompt states the CSV files contain "daily values."
Despite this, 19/24 runs first aggregated daily discharge to monthly
means, then computed mean/std/min/max on those 60 monthly values. This
is a natural language interpretation error: "monthly summary" was read
as "summarize monthly data" rather than "produce a summary table at
monthly resolution of daily data."

All 19 monthly runs produce **identical output values** — the
aggregation logic is deterministic regardless of how the LLM wrote the
Python code.

**Why this matters for the paper:**
- Demonstrates that silent failures are pervasive in the baseline
  condition: the output looks reasonable (correct gauge IDs, plausible
  magnitudes, proper CSV format) but the statistics are computed at the
  wrong temporal resolution.
- The error is invisible unless the reviewer checks n_valid or compares
  against a known reference — exactly the kind of mistake a domain
  scientist might miss on casual inspection.
- Interesting that phi-4 (smallest model) got this right while larger
  models did not — suggests the failure is about prompt interpretation
  habits, not raw capability.

---

## Baseline T2: hand-rolled area weighting (gpt-5 run3)

**Affected runs:**
- `gpt-5` / baseline / `task_02_seasonal_runoff` / run3 → silent failure (idx 28)

**Symptom:** Global area-weighted mean = 0.775 mm/day vs reference
0.746 mm/day (~4% drift). Map spatial pattern is correct but colorbar
includes extreme negative values, washing out land structure.

**Root cause:** GPT-5 built its own `compute_cell_areas_from_1d()`
function (run3.py lines 53-88) that computes cell areas from lat/lon
edges using R² × Δ(sin φ) × Δλ on a perfect sphere. The E3SM grid
has been regridded from an unstructured mesh, so cell areas don't
exactly match a simple spherical calculation. The validated tool in the
protocol condition uses the model's native `area` variable, avoiding
this mismatch.

**Why this matters for the paper:**
- Shows that even when a baseline LLM gets the science right (correct
  variable, correct period, correct spatial pattern), reimplementing
  utility functions from scratch introduces subtle numerical drift.
- The protocol condition avoids this entirely: the validated
  `extract_gridded_field` tool handles area weighting internally.

---

## Baseline T2: no land mask — ocean dilution (o4-mini run1)

**Affected runs:**
- `o4-mini` / baseline / `task_02_seasonal_runoff` / run1 → obvious failure (idx 29)

**Symptom:** Global mean = 2.83e-6 mm/s vs reference 6.65e-6 mm/s —
off by a factor of ~2.4. Map is washed out with no visible land
structure.

**Root cause:** o4-mini used `cos(lat)` weighting over **all grid
cells** including ocean (run1.py lines 55-63). QRUNOFF is zero over
ocean, so including ~70% zero-valued ocean cells dilutes the land-only
mean by roughly the ocean/total area fraction: 6.65e-6 × ~0.43 ≈
2.83e-6. The validated tool applies a land mask automatically.

**Why this matters for the paper:**
- Classic baseline failure: the LLM doesn't know about domain-specific
  masking conventions. The code runs, the output looks like a map, but
  the global statistic is physically meaningless.
- A domain scientist would immediately notice the factor-of-2 error
  and the washed-out map → obvious failure.
- Contrasts with the protocol condition where the tool handles masking
  internally, eliminating this failure mode entirely.

---

## Baseline T2: dropping negative-runoff cells (haiku run3)

**Affected runs:**
- `claude-haiku-4-5` / baseline / `task_02_seasonal_runoff` / run3 → silent failure (idx 31)

**Symptom:** Global mean = 0.600 mm/day vs reference 0.574 mm/day
(~4.5% high).

**Root cause:** Haiku run3 filtered out cells with negative QRUNOFF
values before computing the mean (valid cells = 92,849 vs 94,838 in
runs that kept negatives). Negative runoff values are physically
meaningful in E3SM (representing subsurface drainage or numerical
artifacts). Dropping ~2,000 negative cells biases the mean upward.

---

## Baseline T2: histogram instead of map + ocean dilution (haiku run4)

**Affected runs:**
- `claude-haiku-4-5` / baseline / `task_02_seasonal_runoff` / run4 → obvious failure (idx 32)

**Symptom:** Produced a runoff histogram (linear + log scale) instead
of a spatial map. Global mean = 0.244 mm/day — same ocean dilution
pattern as o4-mini run1.

**Root cause:** Two compounding errors: (1) the LLM chose to plot a
histogram rather than a map, failing the task requirement for a spatial
visualization; (2) computed global mean over all grid cells including
ocean zeros, giving 0.244 mm/day vs reference 0.574 mm/day.

---

## Baseline T3: bilinear regridding instead of conservative remapping

**Affected runs:**
- `claude-opus-4-6` / baseline / `task_03_et_benchmark` / runs 1-4 → silent failure (idx 33-36)

**Symptom:** Global mean bias = 0.278 mm/day (simple) vs reference
0.259 mm/day — a ~7% drift. Spatial correlation (0.851) and bias map
patterns match well. All 4 runs produce identical values.

**Root cause:** The baseline script regrids E3SM (1°) onto the MODIS
grid (0.5°) using `xr.DataArray.interp(method='linear')` — bilinear
interpolation. The validated protocol tool uses conservative remapping
(or computes on the model's native grid), which preserves cell-
integrated quantities. Bilinear interpolation smooths sharp gradients
at land/ocean boundaries and in arid-to-wet transitions, systematically
biasing the spatially averaged difference.

The LLM wrote 530 lines of clean, well-structured Python with correct
variable selection, unit conversion, and visualization — yet this one
numerical methodology choice silently corrupts the headline statistic.

**Why this matters for the paper:**
- This is the paper's core argument in action: an LLM can get the
  science conceptually right while making a subtle implementation
  choice that a domain scientist would catch but a code reviewer
  outside the field would not.
- The error is invisible without a reference: the bias map looks
  correct, the correlation is high, the code is clean. Only comparing
  the global mean against a known-good value reveals the drift.
- The protocol condition eliminates this failure entirely — the
  validated tool encodes the correct regridding methodology, so the
  LLM never has to make that choice.

---

## Baseline T3: MODIS unit conversion omitted (gpt-5 run3)

**Affected runs:**
- `gpt-5` / baseline / `task_03_et_benchmark` / run3 → obvious failure (idx 37)

**Symptom:** Global mean bias = 1.914 mm/day (should be ~0.259).
Entire bias map is red — positive bias everywhere.

**Root cause:** The script converts model ET from mm/s to mm/day
(line 162: `* 86400`) but never converts MODIS ET from kg/m²/s
(numerically mm/s). The "bias" is therefore model_mmday − obs_mm/s ≈
1.902 − (−0.01) ≈ 1.91 — essentially the model ET value itself.

---

## Baseline T3: dask `.item()` failure (gpt-5 run4)

**Affected runs:**
- `gpt-5` / baseline / `task_03_et_benchmark` / run4 → obvious failure (idx 38)

**Symptom:** Bias map looks reasonable (correct mixed red/blue
pattern) but CSV metrics are empty and figure title shows "nan".

**Root cause:** `area_weighted_mean()` calls `.item()` on a dask
array (line 80). The dataset was opened with `xr.open_mfdataset()`
which returns lazy dask arrays. Arithmetic operations work lazily
(so the bias field and map are fine — matplotlib triggers `.compute()`
implicitly), but the explicit `.item()` call in the statistics
function is not valid on dask arrays. The exception is caught silently
(line 276-277) and both metrics are set to `np.nan`.

Fix would be one word: `(num / den).compute().item()`.

**Why this matters for the paper:**
- Shows that baseline code generation must navigate not just the
  science but also library-level API quirks (dask vs numpy).
  The validated tools in the protocol condition are tested against
  these edge cases once; every subsequent LLM invocation inherits
  that correctness.

---

## Baseline T3: dask + longitude convention → all-NaN regridding (o4-mini runs 3-4)

**Affected runs:**
- `o4-mini` / baseline / `task_03_et_benchmark` / run3 → obvious failure (idx 39)
- `o4-mini` / baseline / `task_03_et_benchmark` / run4 → obvious failure (idx 40)

**Symptom:** Bias maps are completely blank (coastlines only, no data).
Run3: bias = 0.0, correlation empty. Run4: all metrics empty. Stderr
shows `All-NaN axis encountered` and `invalid value encountered in
scalar divide`.

**Root cause:** Both runs use `xr.DataArray.interp()` on a dask-backed
array to regrid E3SM (1°, 0-360° lon) onto the MODIS grid (0.5°,
-180 to 180° lon). The longitude convention shift via
`(lon + 180) % 360 - 180` followed by `.sortby('lon')` does not work
correctly with dask lazy arrays — the interpolation targets fall outside
the source coordinate range, producing all-NaN output. The blank bias
field then yields bias = 0.0 (from 0/0 or nansum of empty array) and
no valid cells for correlation.

**Why this matters for the paper:**
- Two independent failure modes compound: (1) longitude convention
  mismatch (a domain-specific data-handling detail), and (2) dask
  lazy evaluation semantics. Neither is about the science — both are
  infrastructure details that the validated tool handles internally.

---

## Baseline T3: MODIS unit conversion omitted (haiku runs 2, 4)

**Affected runs:**
- `claude-haiku-4-5` / baseline / `task_03_et_benchmark` / run2 → obvious failure (idx 41)
- `claude-haiku-4-5` / baseline / `task_03_et_benchmark` / run4 → obvious failure (idx 42)

**Symptom:** Global mean bias = 1.679 mm/day (should be ~0.26).
Obs ET range reported as "0.00 to 0.00 mm/day" in stdout — the obs
values are in kg/m²/s (≈1e-5 order) which round to zero at 2 decimal
places.

**Root cause:** The `load_modis_data()` function computes the MODIS
time mean but never converts from kg/m²/s to mm/day. Model ET is
converted (line 73: `* 86400`), so `bias = model_mmday − obs_mm/s ≈
1.679 − 0.000 ≈ 1.679`. Same unit-mismatch class of error as gpt-5
run3.

Same bug, different model — the ILAMB MODIS ET file's unit metadata
(`kg m-2 s-1`) was available in the file but the LLM didn't check it.

---

## Baseline T4: numpy broadcast error in gauge-to-grid matching (opus runs 1, 2, 4)

**Affected runs:**
- `claude-opus-4-6` / baseline / `task_04_streamflow_fdc` / runs 1, 2, 4 → obvious failure (idx 43-45)

**Symptom:** All FDC panels show "No data" / "No data available".
Maps are blank. CSVs are header-only (0 data rows). Every gauge fails
with `operands could not be broadcast together with shapes (360,)
(720,)`.

**Root cause:** The `find_nearest_grid_cell()` function computes
Euclidean distance as:
```python
dist = np.sqrt((model_lats - lat)**2 + (model_lons - lon_adj)**2)
```
where `model_lats` has shape `(360,)` and `model_lons` has shape
`(720,)`. NumPy cannot broadcast these 1D arrays of different lengths
in an element-wise addition. The fix would be `np.meshgrid` to create
2D arrays, or separate `argmin` per axis, or use `np.abs(lats -
lat)[:, None] + np.abs(lons - lon)[None, :]`.

All gauges fail → no simulated discharge extracted → empty FDC data →
PNG produced but with no content.

**Why this matters for the paper:**
- The code produced a deliverable (PNG file exists, passes crash
  detection) but the content is completely empty — a classic
  "silent crash" that only manual inspection catches.
- The gauge-matching step is a common ESM analysis primitive that
  the validated tool library handles correctly. An LLM reimplementing
  it from scratch hits a basic numpy shape-mismatch bug.

---

## Baseline T5: cftime calendar handling failures (opus run4, gpt-5 run2)

**Affected runs:**
- `claude-opus-4-6` / baseline / `task_05_basin_streamflow` / run4 → obvious failure (idx 46)
- `gpt-5` / baseline / `task_05_basin_streamflow` / run2 → obvious failure (idx 47)
- (opus runs 1-3 crashed outright — empty output directories)

**Symptom:** All 6 basin figures show only the observation time series
— no simulated discharge plotted on any figure. All metrics are NaN,
N_points = 0. The scripts catch the error and continue, producing
obs-only figures for a sim-vs-obs comparison task.

**Root cause:** Two different manifestations of the same underlying
issue — E3SM uses a no-leap/365-day calendar (`cftime.DatetimeNoLeap`):
- **Opus run4:** calls `pd.to_datetime(sim_df["time"])` which fails
  because pandas cannot convert `cftime.DatetimeNoLeap` objects.
- **GPT-5 run2:** calls `.to_period()` on a `CFTimeIndex`, which is
  not supported (only standard `DatetimeIndex` has `.to_period()`).

Both scripts needed `cftime_to_datetime()` or
`values.astype('datetime64[ns]')` before passing to pandas.

**Why this matters for the paper:**
- E3SM uses non-standard calendars (no-leap, 365-day) that require
  explicit handling. This is a domain-specific infrastructure detail
  that the validated tool library (`tools/core/e3sm.py`) handles via
  `cftime_to_datetime()`. LLMs writing from scratch assume standard
  datetime conversion will work.
- Both scripts produced deliverables (6 PNGs each), passed crash
  detection, but every figure is functionally useless — obs-only
  plots for a sim-vs-obs comparison task.
- Two frontier models independently hit the same class of bug through
  different API calls, reinforcing that cftime handling is a systematic
  gap in LLM-generated ESM analysis code.

---

## Baseline T6: missing landfrac weighting in area-weighted means (gpt-5 run3)

**Affected runs:**
- `gpt-5` / baseline / `task_06_water_balance` / run3 → silent failure (idx 49)

**Symptom:** Basin-mean P, ET, Q values are all within ~1–2% of the
reference (e.g., Amazon ET: 3.869 vs 3.874 mm/day; Orange Q: 0.247
vs 0.252 mm/day). Figure layout is correct — residual map with basin
outlines on top, per-basin bar charts below. A domain scientist would
not notice the drift without a reference.

**Root cause:** The script correctly finds the model's native `area`
variable (units km²) and converts to m², but never applies `landfrac`
weighting. The reference protocol tool weights by `area × landfrac`,
which excludes the ocean fraction of coastal grid cells. Without
`landfrac`, partial-ocean cells contribute their full area weight
including ocean portions where water balance variables are zero or
near-zero, systematically pulling basin means slightly toward zero.

The effect is small (~1–2%) because ELM land variables are already
zero over pure ocean cells — only coastal cells with partial land
fractions (0 < landfrac < 1) contribute error. This contrasts with
the much larger ocean-dilution errors seen in T2 (no land mask at all,
~60% drift), where MOSART QRUNOFF zeros over the full ocean dominate.

**Why this matters for the paper:**
- Demonstrates a spectrum of "ocean contamination" errors: from
  catastrophic (no land mask, ~60% error in T2) to subtle (missing
  landfrac, ~1–2% error in T6). Both stem from the same conceptual
  gap — not accounting for land/ocean fractions — but the magnitude
  depends on how much ocean area enters the computation.
- The validated tool encodes `landfrac` weighting as a default,
  eliminating even the subtle end of this error spectrum.

---

## Baseline T6: dask `.item()` + shapely TopologyException + subdirectory output (gemini run4)

**Affected runs:**
- `gemini-2.5-flash` / baseline / `task_06_water_balance` / run4 → obvious failure (idx 77, overriding auto-grade of crash)

**Symptom:** Auto-graded as crash because output was written to a
subdirectory (`run4_output/task06_water_balance/`) instead of directly
into `run4_output/`. On inspection: residual map renders with basin
outlines and colorbar, but the bar chart is completely empty and both
CSVs have headers only (all values blank).

**Root cause:** Three compounding errors:
1. **dask `.item()`** — kills all 4 global mean calculations
   ("'item' is not yet a valid method on dask arrays").
2. **shapely TopologyException** — "side location conflict" on 5 of
   6 basin polygons (Amazon, Missouri, Danube, Mekong, Orange),
   preventing basin-mean extraction. The remaining basin (Columbia)
   hits the dask `.item()` bug instead.
3. **Subdirectory output** — the script created a nested subdirectory
   for its outputs, fooling the crash detector.

**Why this matters for the paper:**
- Illustrates how multiple independent failure modes can compound in
  a single run: a dask API issue, a geometry-library edge case, and
  an output-path convention mismatch. The validated tool library
  handles all three internally.

---

## Baseline T6: precipitation unit conversion off by 1000x (haiku run2)

**Affected runs:**
- `claude-haiku-4-5` / baseline / `task_06_water_balance` / run2 → obvious failure (idx 50)

**Symptom:** Precipitation values ~1000x too small (Amazon P = 0.006
mm/day vs expected ~6.0 mm/day). Residuals massively negative (-6.03
mm/day for Amazon). The residual map is almost entirely deep negative
(range -21 to 0 mm/day). ET and Q are correct.

**Root cause:** Line 80: `precip = (rain + snow) * 86.4` instead of
`* 86400`. RAIN and SNOW are in kg/m²/s (= mm/s); conversion to
mm/day requires multiplying by 86400 (seconds per day). The script
used 86.4 — exactly 1000x too small. The comment on line 83 even
states "1 mm/s = 86400 mm/day" but the code above uses the wrong
constant. ET and Q correctly use `* 86400`.

**Why this matters for the paper:**
- A single-character error (missing two zeros) in a conversion
  constant produces physically impossible results. The water balance
  closes at P − ET − Q ≈ −6 mm/day for the Amazon, which is
  obviously wrong to a hydrologist but would pass any purely
  syntactic code review.
- This is the same class of unit-conversion error seen in T3
  (MODIS ET kg/m²/s not converted), but here it's a typo in the
  conversion factor rather than a missing conversion entirely.

---

## Baseline T6: TopologyException in basin polygon clipping (haiku run3)

**Affected runs:**
- `claude-haiku-4-5` / baseline / `task_06_water_balance` / run3 → obvious failure (idx 51)
- (also affects `gemini-2.5-flash` / baseline / T6 / run4 — see above)

**Symptom:** 5 of 6 basins (Amazon, Missouri, Danube, Mekong, Orange)
have identical values: P=1.73, ET=1.05, Q=0.57, Residual=0.11 mm/day.
Only Columbia has distinct values. The bar chart shows all basins at
roughly equal height, which is physically wrong (Amazon P should be
~6 mm/day, Orange ~2.3 mm/day).

**Root cause:** Shapely raises `TopologyException: side location
conflict` when clipping gridded data to 5 of the 6 basin polygons.
The basin polygon GeoJSON likely has self-intersecting or otherwise
invalid geometry at the conflict coordinates. The script catches the
exception and falls back to a default value (likely the global or
some partial-domain mean), giving identical numbers for all failed
basins. Only Columbia's polygon is clean enough to clip successfully.

**Why this matters for the paper:**
- Basin polygon handling is a common GIS operation in hydrology.
  Real-world basin boundaries from databases like HydroSHEDS or GRDC
  often have topology issues that require `buffer(0)` or
  `make_valid()` preprocessing. The validated tool library uses a
  point-in-polygon approach on the native grid rather than shapely
  clip operations, sidestepping this failure mode entirely.
- Two different models (haiku, gemini) hit the same TopologyException
  on the same 5 basins, confirming it's a data issue that any naive
  shapely-based approach will encounter.

---

## Baseline T7: TopologyException kills 5/6 basins (opus run3)

**Affected runs:**
- `claude-opus-4-6` / baseline / `task_07_integrated_diagnostic` / run3 → obvious failure (idx 52)

**Symptom:** Only Columbia has P/ET/Q data; the other 5 basins are all
NaN. Streamflow metrics (volume bias, Wasserstein distance) work for
all 6 basins because those use point extraction at the gauge, not
polygon clipping. Bar chart shows only Columbia bars.

**Root cause:** Same TopologyException as haiku/gemini T6 — shapely
polygon clipping fails on 5 of 6 basin polygons. The gridded
component (P/ET/Q basin means) requires spatial clipping, which fails,
but the streamflow component uses nearest-grid-cell extraction, which
works independently.

---

## Baseline T7: different grid cell matching → streamflow bias drift (opus run4)

**Affected runs:**
- `claude-opus-4-6` / baseline / `task_07_integrated_diagnostic` / run4 → silent failure (idx 53)

**Symptom:** Gridded water cycle components (P, ET, Q) match the
reference to <0.01% for all 6 basins. Figures are publication-quality
(bar charts, radar diagnostic, bias heatmap, water balance panels).
However, streamflow volume bias diverges significantly for some basins:
Missouri 283% vs reference 200% (83 pp), Danube 38% vs 22% (16 pp).
Other basins are close (Amazon −31% vs −32%, Orange 761% vs 764%).

**Root cause:** The baseline script's nearest-grid-cell matching for
MOSART discharge lands on a different cell than the validated tool for
some gauges. Missouri: baseline matched to index (262, 168) ≈
(41.25°N, 95.75°W); reference matched to (263, 167) ≈ (41.75°N,
96.25°W) — one cell off in both lat and lon. On a routed river
network, one grid cell difference at a gauge location can mean
matching to a different tributary or a point upstream/downstream of
the actual outlet, producing substantially different discharge.

**Why this matters for the paper:**
- This is an archetypal silent failure: the output looks polished and
  physically plausible, the gridded analysis is essentially perfect,
  but a subtle methodological choice (Euclidean nearest-cell vs the
  validated tool's river-network-aware matching) silently corrupts the
  streamflow comparison for specific basins.
- A domain scientist reviewing the figures would not catch this
  without comparing against a known reference — the 283% Missouri
  bias is large but plausible given the known E3SM discharge biases.
- The protocol tool encodes river-network-aware gauge matching,
  eliminating this class of error without requiring the LLM to
  understand MOSART's routing topology.

---

## Baseline T7: inconsistent unit conversion — P/ET in mm/s, Q in mm/day (gpt-5 run2)

**Affected runs:**
- `gpt-5` / baseline / `task_07_integrated_diagnostic` / run2 → obvious failure (idx 54)

**Symptom:** Bar chart shows model P and ET as essentially invisible
(~7e-05 and ~4e-05) while obs P and ET are in mm/day (~6.0 and ~3.5).
Model Q is correctly in mm/day (~2.16). Bias columns in the CSV
compute mm/day − mm/s, giving meaningless values (e.g., Amazon
P_bias = −6.18 mm/day). Streamflow extraction failed for all 6
basins (cftime calendar mismatch).

**Root cause:** The script converted QRUNOFF from mm/s to mm/day
(×86400) but left RAIN+SNOW (P) and QVEGE+QVEGT+QSOIL (ET) in their
native mm/s units. The underlying values are correct — multiplying
by 86400 recovers values within 0.5–1.5% of the reference — but the
inconsistent conversion makes the CSV and figures meaningless.

**Why this matters for the paper:**
- Unit conversion is the single most common class of error across
  all baseline runs (T3 MODIS kg/m²/s, T6 haiku ×86.4, and now T7
  partial conversion). The validated tools encode correct unit
  handling once; LLMs writing from scratch must get it right for
  every variable independently.

---

## Baseline T7: cftime calendar + output path errors (gpt-5 run3)

**Affected runs:**
- `gpt-5` / baseline / `task_07_integrated_diagnostic` / run3 → obvious failure (idx 55)

**Symptom:** Bar chart shows reasonable model vs obs comparison for
P/ET/Q, but streamflow failed for all 6 basins ("No overlapping
streamflow period"). Radar charts are mostly empty (single spokes).
CSVs were saved to `./output/` instead of `run3_output/`, so the
quantitative deliverables are missing from the expected location.

**Root cause:** Two issues: (1) cftime `DatetimeNoLeap` vs pandas
`Timestamp` comparison fails when the script tries to find overlapping
time periods for streamflow data — same calendar-handling gap seen in
T5. (2) Output path hardcoded to `./output/` for some files while
correctly using the run output directory for others, so CSVs are lost.
