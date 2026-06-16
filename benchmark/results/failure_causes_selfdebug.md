# Failure Causes — Self-Debug Runs

Diagnostic notes for the post-self-debug pass of the benchmark. Scope:
artifacts that remained undetermined or failed *after* the self-debug loop,
i.e. after each crashed single-shot run was given up to 3 rounds of
error-driven revision. The original single-shot failure catalog is in
`failure_causes.md`; this file only documents new issues surfaced by the
debug versions.

Reference run for numerical comparison: `claude-opus-4-6` / protocol / run2.

---

## Claude Haiku 4.5 protocol T6: workflow drops runoff-subtraction step

**Affected runs:**
- `claude-haiku-4-5-20251001` / protocol / `task_06_water_balance` / run1 → obvious
- `claude-haiku-4-5-20251001` / protocol / `task_06_water_balance` / run2 → obvious
- `claude-haiku-4-5-20251001` / protocol / `task_06_water_balance` / run4 → obvious
- (run3 is a genuine success — see "Resolved" section below)

**Symptoms:**
- `basin_residual.csv` values are two orders of magnitude larger than the
  reference and uniformly positive:

  | Basin   | Reference    | Haiku run1/2/4 |
  |---------|--------------|----------------|
  | Amazon  | -2.95e-07    | 2.48e-05       |
  | Missouri| 7.50e-07     | 4.05e-06       |
  | Orange  | 9.80e-08     | 3.02e-06       |

- Spatial residual field is essentially `P - ET` instead of `P - ET - Q`,
  so wet regions show large positive imbalance across all ocean-bound
  basins.

**Root cause:** The workflow assembles P-field, ET-field, Q-field, then
calls `compute_spatial_bias(P, ET)` → output labelled as the water-balance
residual. The final runoff-subtraction step is missing:

```yaml
# Run 3 (correct) has this extra step; runs 1/2/4 do not:
- id: compute_residual_final
  tool: compute_spatial_bias
  params:
    field_a: ${compute_residual_field.outputs.bias_file}   # P - ET
    field_b: ${extract_runoff_field.outputs.field_file}    # Q
  outputs:
    bias_file: water_balance_residual.nc                   # P - ET - Q
```

Without this step, the "residual" is just the evapotranspiration deficit
(P - ET), which over most watersheds is strongly positive (reflecting the
runoff that leaves the basin, not storage change).

**Why it survived self-debug:** The original single-shot run crashed for
an unrelated reason; self-debug fixed that crash with a minimal patch
(add a missing input, rename a variable), but never touched the overall
workflow structure. The residual-as-P-ET bug was present in the
single-shot artifact and carried into the debug version unchanged, so no
traceback ever pointed the model at the missing step.

**Paper relevance:** Illustrates a compositional failure mode that
self-debug cannot recover. The artifact *executes* and *produces a named
water-balance residual file*, but the formula is incomplete. An execution
trace alone cannot flag this — numerical comparison against a reference,
or a domain-aware post-condition check (e.g. "mean |residual| ≪ |P-ET|"),
is required.

## T2 baseline silent failures: shared ~0.3 % bias, identical across 7 runs

**Affected runs (all baseline, all graded silent):** gpt-5 r1/r2/r4, o4-mini r2/r4,
gemini-2.5-flash r3/r4 — plus phi-4 r4 as a distinct outlier.

**Symptom:** Every one of the 7 non-phi runs produces a climatological mean QRUNOFF
field whose global mean is **exactly 1.003× the reference** (6.665e-06 vs 6.646e-06
mm/s). Yet only ~31 % of land pixels agree within 1 % — so the *mean* is almost
right but the *spatial pattern* is measurably shifted. Identical ratio and identical
pixel-agreement fraction across 7 independent runs from 3 different models is not
coincidence.

**Likely cause:** Free-generated code computes the 5-year climatology as an unweighted
average of 60 monthly means, whereas the reference workflow's `time_average` tool
length-weights by days-in-month (so Feb counts less than Jul). The ~0.3 % bias and
the spatial-pattern drift are the signature of this weighting mismatch: globally
near-null because days-in-month biases are small, but regionally visible because
seasonality amplitudes differ by latitude.

**Paper relevance:** A second compositional failure mode that self-debug cannot see.
Output executes, plots render, global mean looks right to two digits — but a downstream
consumer doing zonal statistics or interannual anomalies would inherit a small,
systematic bias they'd have no reason to suspect. Protocol mode avoids this because
the validated `time_average` tool always length-weights.

**Phi-4 r4 is different:** mean ratio 0.864, only 0.5 % pixel agreement. This is a
distinct bug (likely a shorter time subset or a different masking choice), not the
monthly-weighting issue above.

---

## T3 baseline obvious failures: all-NaN bias field from MODIS regrid failure

**Affected runs:** gemini-2.5-flash r2, o4-mini r1, o4-mini r2 (all baseline).

**Symptom:** The PNG renders a blank world map with titles like
"Global Mean Bias: nan, Spatial R: nan". The bias NetCDF has 0 % finite pixels.

**Root cause (gemini r2, confirmed by inspection):** MODIS ET climatology was
regridded to 0.5° but ended up with 0 % valid pixels — every cell NaN. The model
field was fine (~35 % finite, matching the land fraction). Subtracting the two
therefore produced an all-NaN bias. The downstream plotting and statistics code
then silently computes NaN means, the plot skips the colormap entirely, and the
text labels show "nan mm/day". The workflow *executes successfully* at every step
because each stage accepts NaN arrays — the corruption propagates without raising.

**Pattern across 3 runs:** Different models, different filenames, same endpoint.
The free-generated code apparently tries something like
`regridded = np.where(mask, modis_raw, np.nan)` or a `xesmf`/interpolation call that
returns all-NaN when lat/lon coordinates don't align. Self-debug can't catch it
because there is no crash, only silent NaN propagation.

**Paper relevance:** Third compositional failure mode invisible to execution traces.
Unlike the T2 weighting bias (numerically close, spatially wrong), this one blows the
entire output into NaN but still produces a file and a figure. A numerical post-check
("fraction of finite pixels ≥ 20 %") would catch it trivially; self-debug's
traceback-driven loop cannot.

**T3 silent failures (4 runs):** haiku r1, gemini r4, gpt-5 r1, gpt-5 r2 all render
realistic bias patterns with plausible global means. Each one is wrong for a
*different* reason — this is the paper's central point about free-generated code:

| Run | Reported bias | Diagnosis |
|-----|--------------|-----------|
| gem r4  | 0.278 mm/day | essentially correct (0.273 ref); 2 % difference due to slightly different land mask — closest to reference |
| gpt r2  | 0.281 mm/day | essentially correct; same land mask as ref, ~3 % numerical drift |
| gpt r1  | 0.111 mm/day | **ocean-fill-zero bug**: bias field is 100 % finite (ref is 19.6 % finite, land only). Land pixels actually mean 0.302, but ocean zeros dilute the global mean to 0.111. The saved NetCDF and on-screen global-mean label are both wrong |
| haiku r1 | 1.68 mm/day | **all-positive bias field** (range 0.04 → 5.37, no negatives) with mean ~4 × the reference's \|bias\|. Consistent with taking `abs()` of bias, or computing RMSE-per-pixel, or dropping the sign somewhere in the pipeline. Bias pattern is visually plausible but magnitude and sign information are lost |

Of these, gpt r1's zero-fill bug is the most insidious: a domain scientist would read
"Global mean bias: 0.11 mm/day" and conclude the model is surprisingly accurate, when
the true land-only bias is 0.30. Protocol mode avoids this because the shared bias
tool returns NaN over ocean and the aggregation routine ignores NaN.

---

## T4 baseline silent failures: Wasserstein-distance normalization mismatch

**Affected runs (silent):** opus r3, gpt-5 r1/r2/r3/r4, o4-mini r1, haiku r1/r4.
**Obvious (separate):** gemini r1 (top-row FDC panels missing the simulated red line —
only observed discharge plotted), o4-mini r4 (FDC panels rendered on log-log axes
without any recognisable discharge pattern).

**Symptom:** Every run renders a plausible 6-gauge FDC comparison and writes a
per-gauge metrics CSV. The CSV's Wasserstein column, however, is numerically
*anti-correlated* with the reference (r ≈ −0.19) and 2–3 orders of magnitude
larger in absolute value:

  | Gauge (obs mean, m³/s) | Reference WD | gpt-5 r1 WD |
  |------------------------|--------------|-------------|
  | 1134100 (Niger)        | 1.06         | 795         |
  | 1159100 (Orange)       | 7.58         | 2389        |
  | 2646200 (small gauge)  | 0.333        | 11 776      |

**Root cause:** The reference workflow normalizes Wasserstein distance by the
observed mean discharge (dimensionless, comparable across gauges). Free-generated
code computes raw Wasserstein in m³/s. Because small gauges can have large *raw* WD
relative to their mean flow but small *normalized* WD (and vice versa for big
rivers), the ordering of gauges gets inverted — a gauge the reference flags as a
good fit may look like the worst fit in the baseline run.

**Why self-debug can't see it:** No exception raised. The FDC panels themselves are
drawn correctly from the daily discharge; only the summary metric is mis-defined.

**One run gets it right:** haiku r3 uses the same normalization convention as the
reference and matches within 6 % median error, corr=0.86 — classified `correct`.
That haiku produced both the normalized (r3) and unnormalized (r1, r4) versions
across its own 4 runs shows the divergence isn't a model limitation but free-code
instability: same model, same prompt, different days → different metric definition.

**Paper relevance:** Distinct from the T2/T3 bugs — not a wrong *value*, a wrong
*definition*. The plots look fine; a reader comparing the Wasserstein column
across models would silently draw wrong conclusions about which gauges are
well-simulated.

---

## T5 baseline silent failures: NaN-handling drift + reference-tool bug exposed

**Affected runs (silent):** opus r1/r2/r3, gpt-5 r1/r3/r4, gemini r3/r4, haiku r1
(9 runs). Obvious: o4-mini r1/r3, haiku r4.

**Symptom:** Every silent run renders a map + monthly discharge comparison that
looks visually correct. The per-gauge NSE values cluster across unrelated runs
from multiple models — six runs (opus r1/r3, gemini r4, gpt-5 r1/r3/r4) produce
*bit-for-bit identical* NSEs to six decimals; two more (haiku r1, gemini r3)
converge on a different but internally-consistent set.

**Initial hypothesis (wrong):** We first read this as "convergent wrong answers" —
free-gen scripts all making the same plausible-but-incorrect default choice.
Inspecting the actual code for opus r1 and gpt-5 r1 disproved this. Both scripts
aggregate daily observations to monthly means before computing NSE:

```python
obs_monthly = obs_df.resample("MS").mean()
merged = sim_monthly.join(obs_monthly, how="inner")
```

This is the hydrologically-correct apple-to-apple comparison: monthly-mean sim
vs monthly-mean obs.

**Actual root cause — a bug in the reference tool.** The validated
`compute_metrics` tool, prior to the fix on 2026-04-14, aligned sim and obs via
plain index intersection:

```python
common_dates = sim.index.intersection(obs.index)   # BUG
```

With monthly sim (timestamps on MS) and daily obs (every calendar day), this
silently kept only the daily obs values landing on day-1-of-each-month — a
single day's discharge per month, not a mean. Verified numerically for the
Orange gauge:

  | Method                                       | n   | Orange NSE |
  |----------------------------------------------|-----|------------|
  | `sim.index.intersection(obs.index)` (old ref)| 58  | **−5.622** |
  | `obs.resample("MS").mean()` (cluster A)      | 59  | **−12.824**|

The reference was the one getting a wrong answer; Cluster A was correct.

**Fix applied:** `tools/analyzers/compute_metrics.py` now infers sampling
frequency from the median index step of each series; when they differ by >1 day,
the finer-resolution series is resampled to the coarser frequency's mean before
intersecting. After the fix, all protocol T5 runs produce numerically-sensible
NSEs (e.g. Orange = −12.82, matching Cluster A), and the self-debug manual
labels no longer rest on a silently-misaligned reference.

**Why baseline "silent" labels survive the fix.** With the corrected reference,
Cluster A's Orange NSE matches bit-for-bit (−12.824 = −12.824), but other
gauges still disagree (Missouri: baseline −93.4 vs fixed-ref −45.9; Danube:
baseline −0.61 vs fixed-ref +0.08). The divergence is no longer about
alignment — it's about **NaN-handling when aggregating daily obs**. Free-gen
scripts call `resample("MS").mean()` with pandas' default `skipna=True`, which
silently computes a mean over whatever days are present in each month, producing
different monthly values for gauges with gappy records (Missouri, Danube).
The fixed reference tool does the same by default but over a slightly different
obs file (post-matched-gauges CSV rather than raw USGS), so the NaN patterns
don't coincide.

**Paper relevance:** T5 is the cleanest example of a failure mode we had not
anticipated — a bug embedded in the validated reference tool, surfaced only
because six independent free-gen scripts converged on the *correct* answer and
disagreed with the reference. Without the cross-check provided by baseline
runs, the reference's day-1-of-month artifact would have shipped unnoticed.
The lesson for the paper: the protocol-first story isn't "validated tools
never fail" — it's that (a) validated tools fail less often and more
recoverably than free-gen code, and (b) cross-mode benchmarking against
varied free-gen implementations is itself a useful test of the tool library.
Cluster A is now re-interpreted as "hydrologically reasonable but not
bit-for-bit reproducible" rather than "convergent wrong answer."

**Obvious (separate):**
- o4-mini r1: plot canvas blank, metrics all NaN.
- o4-mini r3: only 6 annual data points (not monthly); NSE ~25–80× off reference
  across basins — likely aggregated daily to yearly instead of monthly.
- haiku r4: only a single line plotted per basin panel — observation series
  missing or overplotted by sim.

---

## T6 baseline near-matches and Haiku unit bug

**Affected runs (silent — near-match to reference):** opus r1, gemini r3,
gpt-5 r1/r4, o4-mini r2 (5 runs). **Correct:** opus r2, opus r4. **Obvious:**
haiku r1, haiku r4. **Also silent (different cluster):** o4-mini r1/r3/r4.

**Near-match symptom:** Five free-gen runs produce basin-mean residuals that
are bit-for-bit identical to reference on Amazon, Columbia, Danube, and Orange
(to 8 decimals), but differ on Missouri and Mekong by small amounts
(Missouri 0.0649 vs ref 0.0648 mm/day; Mekong 0.0175 vs ref 0.0170). The
pattern is consistent enough across unrelated models to suggest a shared
cause: these two basins have patchier obs coverage (gauge mask clipping) than
the other four, so a minor difference in how each script handles
partial-basin grid cells propagates only to those basins. All five still
fail bit-for-bit against the reference.

**o4-mini cluster (r1/r3/r4):** A distinct residual set where even Amazon
differs (−0.0433 vs ref −0.0255). All basins differ by 50-100 % from the
reference. Consistent with a different basin mask or upstream data-selection
choice made earlier in the pipeline. Not catastrophic — residuals are still
small, just shifted uniformly.

**Haiku unit bug (obvious):** Both haiku runs show precipitation values
approximately 1000× smaller than expected (Amazon P = 0.006 mm/day vs ~6).
ET and Q are in correct mm/day, so the computed residual
(P − ET − Q) is dominated by −(ET + Q) ≈ −5.8 mm/day across all basins.
The bar chart renders visibly broken — P bars are invisible next to
full-height ET and Q bars — which is why these are obvious rather than
silent. Root cause is a missed unit conversion on the precipitation field
(kg m⁻² s⁻¹ kept instead of converted to mm/day).

**Paper relevance:** T6 is a cleaner result than T5 — two opus runs are
genuinely bit-for-bit correct, and the silent runs cluster close to the
reference with small, explainable differences. The haiku unit bug is a
classic baseline-vs-protocol failure mode: protocol mode never applies a
unit conversion manually, so it cannot fail this way. The five near-match
runs illustrate that even when free-gen code gets the overall structure
right, small upstream choices (grid-cell masking, partial-cell handling)
are enough to break bit-for-bit reproducibility.

---

## T7 baseline: unit chaos in a multi-variable task

**Affected runs:** opus r2 (silent); haiku r1/r2/r4, gemini r4, gpt-5 r4,
o4-mini r2/r4 (7 obvious).

**Symptom:** T7 asks for an integrated diagnostic combining P, ET, Q, and
streamflow across six basins, with both bar-chart and radar visualisations.
Every baseline run that reached the end produces a CSV whose column units
mix mm/s, mm/day, and m³/s in the same row. The resulting bias columns
read either −99.99 % (model-units ÷ obs-units where the two are 86400× apart
but the script doesn't know) or in the millions-of-percent range (e.g.
o4-mini r2 shows Q_bias = 6,037,547 % because Q_mod is in mm/day but
Q_obs was fetched in mm/s and never converted).

**Why it's almost always obvious here.** T7 requires the model to juggle
three variables with different native units simultaneously. A small
mis-conversion on one of them pushes a bias metric into the millions-of-%
range, which no downstream plot can hide. In contrast, T6 (water balance)
only needs P, ET, Q *in the same model*, so one consistent unit choice
works throughout.

**opus r2 silent failure:** The only near-correct baseline. Values are
consistently mm/day and match the reference pattern for P, ET, Q. The
failure is the same Wasserstein-distance normalization bug documented in
T4 — opus r2 reports Wasserstein = 54 144 (raw m³/s) where the reference
gives 0.32 (normalized by obs mean). Bar chart is plausible; radar chart
has one axis that's ~170 000× too long and a reader who skimmed the radar
would misread relative gauge performance.

**Paper relevance:** T7 is the most demanding free-gen task in the suite,
and the baseline failure rate (17/24 crashed, 7/7 remaining runs obvious)
reflects that. It illustrates that compositional complexity compounds
unit-handling risk: each additional variable is one more chance for a
silent unit mismatch, and in T7 those mismatches become loud because
bias percentages explode. Protocol mode avoids this because unit
conversions happen inside validated tools once per variable and are never
re-derived in a downstream tool.

---

## Resolved — Haiku T6 grading fix

**Resolved (grading fix):** Haiku run3 does include the runoff-subtraction
step and its `water_balance_residual.nc` matches the reference bit-for-bit.
It was initially flagged undetermined because `find_key_nc` was ordered on
the generic pattern `residual`, and Haiku's pipeline emits three
intermediate NCs whose names also contain that substring
(`et_residual.nc`, `precip_residual.nc`, `runoff_residual.nc`). Alphabetic
sort picked `et_residual.nc` (an intermediate ET-minus-observation bias
field) and compared that to the reference's `field_residual.nc`, which
unsurprisingly disagreed. Patched the pattern list in both
`structural_grading.py` and `bitwise_comparison.py` to prefer
`water_balance_residual` / `field_residual` before falling back to the
generic `residual` token; run3 is now auto-graded as success.
