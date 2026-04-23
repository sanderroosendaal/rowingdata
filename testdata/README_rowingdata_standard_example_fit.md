# `rowingdata_standard_example.fit`

Reference **Garmin FIT** activity file produced by **rowingdata** to illustrate the layout and developer fields described in `rowingdata/data/fit_export_spec.json` and `docs/FIT_EXPORT.md`.

## Contents

- **Source data:** `rp3intervals2.csv` via `RowPerfectParser` (RowPerfect RP3 export with per-stroke `curve_data`).
- **Laps:** ` lapIdx` is set from distinct `workout_interval_id` values (**5** intervals → **5** Lap messages + per-interval Records).
- **Sport:** `indoor_rowing` (indoor rowing / erg).
- **Recording strategy:** Stroke-boundary (one Record per stroke, **RecordingStrategy** field ID 10 = `RECORDING_STRATEGY_STROKE_BOUNDARY`). See `docs/FIT_EXPORT.md` "Record message frequency" for details.
- **Developer fields:** Rowing metrics (e.g. DriveLength, StrokeDriveTime, drag, forces, WorkoutState, peak position when present), plus **downsampled** in-stroke **HandleForceCurve** (16 points) and axis metadata **InstrokeAbscissaType**, **InstrokeSampleInterval**, **InstrokePointCount** (fields 90–92).
- **In-stroke mode:** `downsampled` with `instroke_downsample_points=16`, `instroke_abscissa_type=TIME_UNIFORM_MS` (constant `INSTROKE_ABSCISSA_TIME_UNIFORM_MS` in `rowingdata.fitwrite`).

## Regenerate

From the repository root:

```bash
python tools/build_example_standard_fit.py
```

Use this when `fit_export_spec.json` or export logic changes and the example file should be refreshed.
