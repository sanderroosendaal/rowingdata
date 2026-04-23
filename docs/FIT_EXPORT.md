# FIT Export – rowingdata support for Intervals.icu

The rowingdata library exports rowing sessions to Garmin FIT format for use with [Intervals.icu](https://intervals.icu) and other platforms. Here's what we export and how it maps.

Per the [Garmin FIT SDK Activity structure](https://developer.garmin.com/fit/file-types/activity/), FIT files use Activity, Session, Lap, and Record messages. We emit summary data at session level (whole workout), split/interval level (per lapIdx), and record level (per stroke).

## Usage

```python
import rowingdata

row = rowingdata.rowingdata(csvfile="workout.csv")
row.exporttofit("workout.fit", sport="rowing", notes="My workout")
```

Parameters: **fileName** (output path), **notes** (default: "Exported by Rowingdata"), **sport** (e.g. rowing, indoor_rowing), **use_developer_fields** (default: True – include rowing-specific fields when present), **instroke_export** (default: 'off' – see In-stroke curve export below), **instroke_columns** (optional list of curve columns), **instroke_column_map** (optional override mapping), **instroke_downsample_points** (default: 16 for downsampled export; range 2–127), **instroke_abscissa_type** (optional; X-axis semantics for in-stroke curves – see In-stroke abscissa below; `None` = auto), **instroke_sample_interval_ms** (optional override for sample spacing; meaning depends on `instroke_abscissa_type`), **overwrite** (default: True – set False to raise `FileExistsError` if the target file or companion already exists), **garmin_parity_source_fit** (optional; path to a source FIT whose native Workout / WorkoutStep / Split / SplitSummary messages are preserved after Session – see Intervals.icu compatibility below).

**Return value**: `None` in the normal case. When notable conditions occur, returns a dict:
- `instroke_columns_available`: list of column names when in-stroke data is detected but `instroke_export='off'` (allows you to decide whether to re-export with `instroke_export='summary'` or `'companion'`)
- `suggestion`: hint to re-export with instroke_export enabled
- `companion_file`: path to the `.instroke.json` sidecar when `instroke_export='companion'` writes one

## Reference example FIT

The repository includes **`testdata/rowingdata_standard_example.fit`**: a full activity file (multi-lap, rowing developer fields, downsampled in-stroke curve + axis metadata) built from `testdata/rp3intervals2.csv`. See **`testdata/README_rowingdata_standard_example_fit.md`** for how it was produced and how to regenerate it after spec changes (`python tools/build_example_standard_fit.py`).

## Authoritative developer field list (machine-readable)

FIT **developer field IDs**, FIT names, base types, scales, DataFrame column mappings, in-stroke dynamic ID ranges (`summary_start`, `curve_start`), abscissa enum, and related metadata are defined in **`rowingdata/data/fit_export_spec.json`** (shipped with the package). `rowingdata/fitwrite_spec.py` loads and validates it; `rowingdata/fitwrite.py` uses the loaded tuples. Prose tables in this document should stay aligned with that JSON when the standard changes.

## Native vs developer fields

**Native fields** are part of the Garmin FIT SDK (timestamp, distance, heart_rate, position_lat, cycle_length, etc.). Every FIT-capable app understands them.

**Developer fields** are custom fields defined in the FIT file. They need extra metadata; apps may ignore them unless they explicitly support rowingdata's fields.

We export native fields for standard metrics plus developer fields for rowing-specific columns when use_developer_fields=True and the column exists.

## Naming and field choices

- **StrokeDriveTime** – drive phase time (ms). Named for symmetry with StrokeRecoveryTime.
- **DriveLength** – handle distance (meters). Per README: distance traveled by the handle along the longitudinal axis of the boat or erg. For OTW rowing, this is the projection of the handle trajectory on the longitudinal axis (not the full path the hands travel). For indoor rowing, typically handle travel catch-to-finish. Typical range: 1.2–1.5 m. This is a *stroke output* metric.
- **StrokeDistance** – distance traveled during the stroke cycle (meters). Per README: the distance the boat/erg travels during one stroke cycle. Distinct from DriveLength (handle distance). Typical range: 7–12 m for OTW. Uses native FIT field **cycle_length16** (UINT16, scale 100, max 655 m) for interoperability with Garmin Connect, Intervals.icu, and other FIT parsers.
- **DriveLength vs EffectiveLength** – Not duplicates. **DriveLength** is the actual distance the handle traveled during the stroke (~1.2–1.5 m). **EffectiveLength** is an oarlock/rigging metric: the effective lever length (horizontal component from pin to handle, often in cm). It describes rigging geometry, varies slightly stroke-to-stroke, and comes from NK Logbook (Oarlock). Both can appear in the same dataset.
- **Stroke Number** – stored in the native **total_cycles** field (Garmin/ANT+ cycles). Rowing machines report it via ANT+ to Garmin watches; we write it per record since data is per stroke.

**StrokeWork (J)** – work done over one **full stroke cycle** (joules), not drive-phase-only work. The exporter maps the first present DataFrame column among ` WorkPerStroke (joules)` and `driveenergy`. Those names describe the same physical quantity in most pipelines: device-reported joules per stroke, or derived **average power × stroke period** (e.g. `60 * Power / Cadence`). The internal name **`driveenergy`** is historical and can be misread as “drive phase only”; **StrokeWork** is the preferred FIT label for this full-cycle metric.

## Developer fields exported

Field definition numbers (**Dev field ID**) match `rowingdata/data/fit_export_spec.json`. In-stroke curve summary and array fields use dynamic IDs from **`instroke_dynamic`** in that file (default summary from **20**, curve arrays from **60**); in-stroke axis metadata uses **90–92** (see [In-stroke abscissa](#in-stroke-abscissa-x-axis) below).

**Drive and peak force (IDs 4–7):** **Newtons** are preferred for new code and documentation: **` AverageDriveForce (N)`** / **` PeakDriveForce (N)`** → **AverageDriveForceN** / **PeakDriveForceN** (IDs **6** and **7**). **Pounds** (**AverageDriveForceLbs** / **PeakDriveForceLbs**, IDs **4** and **5**) are still written when the corresponding **`...(lbs)`** columns exist—**backward compatibility** only; avoid new pipelines that depend on lbs unless required for legacy tools.

| rowingdata column | FIT field name | Dev field ID | Base type | Scale | Units |
|-------------------|----------------|--------------|-----------|-------|-------|
| DriveLength (meters) | DriveLength | 0 | UINT16 | 100 | m |
| DriveTime (ms) | StrokeDriveTime | 1 | UINT16 | 1 | ms |
| DragFactor | DragFactor | 2 | UINT16 | 1 |  |
| StrokeRecoveryTime (ms) | StrokeRecoveryTime | 3 | UINT16 | 1 | ms |
| AverageDriveForce (N) | AverageDriveForceN | 6 | UINT16 | 10 | N |
| PeakDriveForce (N) | PeakDriveForceN | 7 | UINT16 | 10 | N |
| AverageDriveForce (lbs) | AverageDriveForceLbs | 4 | UINT16 | 10 | lbs |
| PeakDriveForce (lbs) | PeakDriveForceLbs | 5 | UINT16 | 10 | lbs |
| AverageBoatSpeed (m/s) | AverageBoatSpeed | 8 | UINT16 | 100 | m/s |
| WorkoutState | WorkoutState | 9 | UINT8 | 1 |  |
| (metadata, see Record message frequency) | RecordingStrategy | 10 | UINT8 | 1 |  |
| ` WorkPerStroke (joules)` (first match) or `driveenergy` | StrokeWork | 19 | UINT16 | 1 | J |
| catch, catchAngle | Catch | 11 | SINT16 | 10 | deg |
| finish, finishAngle | Finish | 12 | SINT16 | 10 | deg |
| slip | Slip | 13 | SINT16 | 10 | deg |
| wash | Wash | 14 | SINT16 | 10 | deg |
| peakforceangle | PeakForceAngle | 15 | SINT16 | 10 | deg |
| effectiveLength | EffectiveLength | 16 | UINT16 | 100 | m |
| rel_peak_force_pos, PeakForcePositionNorm, `% of Stroke Complete When Peak Force Is Reached` | PeakForcePositionNorm | 17 | UINT16 | 1 | (see below) |
| peak_force_pos, PeakForcePositionAbs | PeakForcePositionAbs | 18 | UINT16 | 100 | m |

**PeakForceAngle** is the oar angle (degrees) at peak force (water / oarlock). **PeakForcePositionNorm** and **PeakForcePositionAbs** describe where along the drive the force maximum occurs (indoor / RP3-style metrics). Do not confuse angle with position along the drive.

- **PeakForcePositionNorm** – UINT16 **0–10000**: ten-thousandths of unity along the drive phase (0 = catch, 10000 = end of drive). FIT `scale` is 1 (Garmin field descriptions allow scale 0–255 only). Source mapping:
  - **rel_peak_force_pos** (RowPerfect / RP3): relative position; values in 0–100 are treated as percent and converted; values in 0–1 are treated as fractions.
  - **PeakForcePositionNorm**: explicit column in the same units (0–1 or 0–100).
  - **`% of Stroke Complete When Peak Force Is Reached`** (ETH export): percent of stroke at peak force.
- **PeakForcePositionAbs** – handle travel from catch to peak force in **meters** (scale 100). **peak_force_pos** from RP3 is often in **centimetres**; values **> 2.5** are divided by 100 to obtain metres; smaller values are assumed already in metres.

Oarlock scalars (catch, finish, slip, wash, peakforceangle, effectiveLength) are exported when present. NK Logbook (Oarlock) uses these columns. See README *Oarlock scalars (OTW rigging)* for definitions. **EffectiveLength** is distinct from **DriveLength**: the former is rigging geometry (effective lever length); the latter is actual handle travel distance.

These have no native equivalents. Apps like Intervals.icu can import them when the developer field descriptions are present.

## Symmetry with two oarlocks

When a rower uses two smart oarlocks (e.g. dual EmPower or Quiske per-side data), oarlock metrics can be reported per side: **port** (left) and **starboard** (right). We export per-side developer fields when both sides are present, and always export summary fields so partial implementations still get useful data.

### Per-side developer fields

Developer field IDs **200–211** (see `rowingdata/data/fit_export_spec.json`, group `oarlock_dual`).

| rowingdata column | FIT field name | Dev field ID | Base type | Scale | Units |
|-------------------|----------------|--------------|-----------|-------|-------|
| catch_port, catchAngle_port | CatchPort | 200 | SINT16 | 10 | deg |
| catch_starboard, catchAngle_starboard | CatchStarboard | 201 | SINT16 | 10 | deg |
| finish_port, finishAngle_port | FinishPort | 202 | SINT16 | 10 | deg |
| finish_starboard, finishAngle_starboard | FinishStarboard | 203 | SINT16 | 10 | deg |
| slip_port | SlipPort | 204 | SINT16 | 10 | deg |
| slip_starboard | SlipStarboard | 205 | SINT16 | 10 | deg |
| wash_port | WashPort | 206 | SINT16 | 10 | deg |
| wash_starboard | WashStarboard | 207 | SINT16 | 10 | deg |
| peakforceangle_port | PeakForceAnglePort | 208 | SINT16 | 10 | deg |
| peakforceangle_starboard | PeakForceAngleStarboard | 209 | SINT16 | 10 | deg |
| effectiveLength_port | EffectiveLengthPort | 210 | UINT16 | 100 | m |
| effectiveLength_starboard | EffectiveLengthStarboard | 211 | UINT16 | 100 | m |

Per-side fields are exported only when both port and starboard columns exist for that metric.

### Summary fields (Catch, Finish, etc.)

The non-suffixed fields (Catch, Finish, Slip, Wash, PeakForceAngle, EffectiveLength) serve as summary values so developers who implement only the partial standard still get meaningful data:

| Scenario | Catch (and other oarlock scalars) |
|----------|-----------------------------------|
| Single oarlock (port or starboard) | That side's value |
| Dual oarlock (both present) | Average of Port and Starboard |
| Dual oarlock (one side missing) | The side that is present |

This keeps backward compatibility and gives partial implementers a representative per-stroke value. Full implementers can use the per-side fields for symmetry analysis (e.g. catch angle imbalance).

## Record message mapping (native fields)

| rowingdata column | FIT field | Notes |
|-------------------|-----------|-------|
| TimeStamp (sec) | timestamp | UTC; relative timestamps combined with row_date |
| cum_dist or Horizontal (meters) | distance | Cumulative meters (FIT scale 100) |
| Cadence (stokes/min) | cadence | Strokes/min; omitted if zero |
| HRCur (bpm) | heart_rate | Clamped 0–255 |
| Power (watts) | power | Clamped 0–65535 |
| Stroke500mPace (sec/500m) | enhanced_speed | Converted to m/s via 500/pace |
| latitude | position_lat | Degrees; omitted if invalid |
| longitude | position_long | Degrees; omitted if invalid |
| Stroke Number or row index | total_cycles | Cumulative stroke count; may repeat across records in GPS-update mode |
| StrokeDistance (meters) | cycle_length16 | Native UINT16, scale 100, max 655 m |

## Record message frequency

Different vendors generate FIT Record messages at different frequencies, and **consumers MUST be agnostic** about this distinction:

- **Stroke-boundary records** (rowingdata, rowing-specific devices): One Record message per stroke cycle. Each Record represents a complete stroke. Stroke-specific metrics (cadence, stroke distance, drive time, etc.) describe that specific stroke. GPS position may be interpolated or repeated across strokes. **Required when in-stroke curve data is present** (InstrokeAbscissaType fields 90-92).

- **GPS-update records** (Garmin watches, multi-sport devices): Records generated when GPS position updates are available (typically roughly once per second, but not at fixed intervals). Stroke metrics may be averaged, interpolated, or represent the most recent completed stroke. GPS position is precisely measured at each record time. **Cannot include in-stroke curve data** since curves require stroke boundaries.

### Consumer requirements

To correctly handle both approaches, consumers MUST:

- **Not assume** a 1:1 correspondence between Record messages and strokes
- **Not interpolate** stroke-specific developer fields (DriveLength, StrokeDriveTime, Catch, Finish, oarlock angles, etc.) between records—these describe discrete stroke events, not continuous phenomena
- **Detect stroke occurrences** by monitoring changes in `total_cycles`, not by counting records. When `total_cycles` changes between consecutive records, at least one stroke completed in that interval. If the change is >1, multiple strokes occurred but per-stroke data for intermediate strokes is unavailable.
- **Calculate stroke rate** from the native `cadence` field (strokes/min), not from record message frequency
- **Understand GPS-update limitations**: When records are generated at GPS updates rather than stroke boundaries, stroke timing is approximate (occurred sometime between records), and fast rowing may cause `total_cycles` to skip values

### Recording strategy metadata (optional)

To help consumers optimize parsing and provide explicit documentation of producer intent, the **RecordingStrategy** developer field (ID 10, UINT8) indicates the recording approach:

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | Unknown | Recording strategy unspecified (default for backward compatibility; consumers must handle both approaches) |
| 1 | StrokeBoundary | One Record per stroke cycle (rowingdata default; required for in-stroke curve data) |
| 2 | GPSUpdate | Records generated at GPS position updates (event-driven, roughly ~1 Hz but irregular) |

This field is **optional**. When omitted or zero, consumers must not assume any particular strategy and should monitor changes in `total_cycles` to detect when strokes occur. When present, it allows consumers to optimize (e.g., in stroke-boundary files, each record is exactly one stroke) but is not required for correct parsing.

**Constraint**: In-stroke curve data (developer fields 90-92 for axis metadata, 60+ for curve arrays) can only appear when RecordingStrategy is StrokeBoundary (1) or Unknown (0 with stroke-boundary semantics), since curves are inherently per-stroke.

**Rationale**: Different devices have legitimate reasons for each approach. Stroke-boundary recording is ideal for rowing-specific devices that detect stroke events in real-time, providing precise per-stroke metrics and enabling in-stroke curve export. GPS-update recording is standard for multi-sport watches that maintain accurate GPS positioning and cannot always detect sport-specific events in real-time. Requiring consumer agnosticism ensures the rowing data ecosystem remains interoperable across diverse hardware.

## Session, Lap, and Event messages

- **Session** – One message for the whole workout (total_distance, total_calories, avg_heart_rate, max_heart_rate, avg_cadence, avg_power). Start/end position in degrees when valid.
- **Lap** – One Lap message per interval when the data has multiple unique `lapIdx` values (supports both ` lapIdx` and `lapIdx` column names). Each Lap has per-interval distance, elapsed time, total_calories, avg HR, max HR, avg cadence, avg power. **Elapsed and timer time** use **wall-clock** duration from the **first** stroke of the interval to the **first** stroke of the **next** interval (or the **last** stroke of the session for the final interval), matching typical Garmin FIT Lap semantics and avoiding zero-duration laps when an interval contains only one stroke. Per-interval avg HR, cadence, and power use **work strokes only** (WorkoutState 1,4,5,6,7,8,9); rest strokes (WorkoutState 3) are excluded from those averages. If `lapIdx` is missing or all values are the same, one Lap for the whole session.
- **Event** – Lap boundaries are marked with Event messages (Event.LAP, EventType.START) before each lap's records. Timer start/stop events bracket the activity.

## Dependencies

```bash
pip install fit-tool
```

## Timestamps and structure

- Relative timestamps (below year 2000) are combined with row_date.
- FIT timestamps: milliseconds since Garmin epoch (1989-12-31 UTC).
- File structure: File ID, Activity, Event (start), Session, Developer data (if any), then for each interval: Event (lap start), Lap, Record messages for that interval, then Event (stop). If there is only one interval: one Lap, then all Records.

## FIT import (`FITParser`)

[`rowingdata.FITParser`](rowingdata/otherparsers.py) reads **Record** messages into a DataFrame. **` lapIdx`** is derived from **Lap** messages: each lap contributes **start_time** (or **timestamp** if **start_time** is absent), laps are sorted by time (then **message_index** when present), and each stroke gets the index of the **last** lap whose start is ≤ the record timestamp (0-based). Strokes before the first lap start use index **0**; there is no reliance on global message order, so laps listed before the first **Record** still bracket strokes correctly. If no lap times can be parsed, **` lapIdx`** is all zeros. **`rowingdata.fit_transcode.data_frame_from_garmin_fit`** uses **`FITParser`** internally so Garmin/ORM transcoding keeps the same **` lapIdx`** semantics.

## Intervals.icu compatibility

We structure the FIT file for compatibility with [Intervals.icu](https://intervals.icu):

- **Lap messages (default)**: We emit Lap messages for intervals (one per `lapIdx`). **By default** we do not emit Garmin **Split** messages; the column `lapIdx` is a rowingdata and Concept2-derived convention for interval indexing (not a Garmin FIT field). For Intervals.icu import behaviour this remains the default; see **Splits vs Laps** below and [issue #63](https://github.com/sanderroosendaal/rowingdata/issues/63).
- **Optional Split parity (Garmin / OpenRowingMonitor)**: When transcoding from a source FIT that already contains native **Workout**, **WorkoutStep**, **Split** (FIT global message number **312**), and **SplitSummary** (**313**), pass **`garmin_parity_source_fit=<path>`** to **`rowingdata.fitwrite.write_fit`** or **`rowingdata.rowingdata.exporttofit`**, or use **`rowingdata.fit_transcode.transcode_garmin_fit_to_rowingdata`**. Those native messages are re-read with **fitparse** and re-emitted after the **Session** message via **`rowingdata.fit_garmin_bridge`**; per-stroke **Record** data and **rowingdata** developer field definitions still come from the DataFrame (ORM-style record fields are mapped in **`fit_transcode.data_frame_from_garmin_fit`**). This gives viewers (e.g. fitfileviewer.com) Split + Lap + rowingdata-spec strokes; message ordering may still differ from a pure Garmin export.
- **Summary First ordering**: Each Lap message precedes the Record messages it summarizes. Garmin and OpenRowingMonitor use Summary Last (Records before Lap); we use Summary First for compatibility with Intervals.icu. **We need to confirm with Intervals.icu** whether they support Summary Last and whether switching would improve interoperability. See [issue #61](https://github.com/sanderroosendaal/rowingdata/issues/61).

### Splits vs Laps (Garmin practice and linkage)

Community analysis of Garmin native indoor rowing FIT files (see [issue #63](https://github.com/sanderroosendaal/rowingdata/issues/63), including feedback from OpenRowingMonitor) indicates:

- **Splits as intervals**: Garmin **Split** messages usually correspond 1:1 to **workout steps** (i.e. the logical intervals in the workout). Tools such as OpenRowingMonitor emit splits on that model because Garmin’s toolchain handles it well.

- **Split–Lap linkage is not absent in the wild**: Splits are summary messages and can be tied to laps. Observers report bidirectional linkage via native (sometimes undocumented) fields—for example a split field indicating **which lap the split starts on** (described in discussion as a “first lap index” style field) and lap fields such as a **workout step index** pointing back to the workout step (and thus the split, when steps and splits align 1:1). Exact field names and SDK exposure vary; rowingdata does not emit these yet.

- **Splits without full linkage**: A split message can still function as a **standalone per-interval summary**; consumers can parse and display interval totals from splits alone without recomputing from underlying Records. That can be preferable for some pipelines (e.g. where post-processing of stroke-level rowing data is weak).

- **Related lap fields**: Discussion of undocumented lap fields that make lap summaries closer to split-style summaries appears in [markw65/fit-file-writer#14](https://github.com/markw65/fit-file-writer/issues/14).

**Rowingdata today**: Default export remains Lap-only + Summary First for Intervals.icu. **Optional** Split + Workout step parity is implemented as above (`garmin_parity_source_fit`). Documented linkage fields on laps/splits are still not generated from rowingdata alone; full Garmin ordering (Summary Last, etc.) is not replicated unless contributed.
- **sub_sport**: For rowing activities we infer from data when possible:
  - If `sport` is explicitly `indoor_rowing` or `indoor rowing` → `indoorRowing` (indoor/erg)
  - If `sport` is explicitly `water` → generic (on-water)
  - If `sport` is `rowing` or default → GPS present → generic (on-water OTW); no GPS → `indoorRowing`

## In-stroke abscissa (X-axis)

Y-only curve arrays are ambiguous (time vs handle distance vs oar angle). When **`instroke_export`** is **`'downsampled'`** or **`'full'`**, we add three per-Record **developer fields** (IDs 90–92) whenever curve data is written into the FIT file:

| FIT field name | Base type | Meaning |
|----------------|-----------|---------|
| InstrokeAbscissaType | UINT8 | Enum (see below). |
| InstrokeSampleInterval | UINT16 | Spacing between samples; **interpretation depends on InstrokeAbscissaType** (see below). |
| InstrokePointCount | UINT8 | Number of points in each exported curve array for that record (≤127). |

**InstrokeAbscissaType** values (module constants in `rowingdata.fitwrite`: `INSTROKE_ABSCISSA_*`):

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | UNKNOWN | Abscissa not specified. |
| 1 | TIME_UNIFORM_MS | Uniform time sampling; **InstrokeSampleInterval** = milliseconds between samples. Default when ` DriveTime (ms)` is present: `drive_ms / (point_count - 1)` rounded. |
| 2 | HANDLE_DISTANCE_UNIFORM_M | Uniform spacing along handle travel (metres); **InstrokeSampleInterval** uses scale documented in field (0.01 m steps if using integer ms-style storage—consumers should follow exporter notes). |
| 3 | OAR_ANGLE_UNIFORM_DEG | Uniform oar angle spacing; **InstrokeSampleInterval** in 0.1° if documented with scale. |
| 4 | NORMALIZED_DRIVE_0_1 | Dimensionless 0–1 along drive; interval is step size in 1/10000 if using integer storage. |

**Oar angle and Catch / Finish.** For on-water curves sampled uniformly in **oar angle**, the meaningful domain is **[θ_catch, θ_finish]** in the same angular convention as the stroke scalars **Catch** and **Finish** (developer fields on the same Record). Conceptually, sample index `k` maps to θ_k = θ_catch + k × Δθ with Δθ derived from (θ_finish − θ_catch) and **InstrokePointCount**, or equivalently from **InstrokeSampleInterval** if that is how spacing is defined. Exporters should take θ_catch / θ_finish from those fields when present.

**Missing Catch / Finish and UNKNOWN.** If **Catch** or **Finish** is missing (or cannot be aligned with the curve), treat the abscissa as unspecified: use **InstrokeAbscissaType = UNKNOWN (0)** and interpret the curve as **shape-only**—useful for relative shape, clustering, or normalization, not for plotting against absolute oar angle without additional metadata (e.g. companion `x[]` or future start/end fields).

**Consistency (not enforced).** In angle-based or time-based modes, **InstrokeSampleInterval**, **InstrokePointCount**, and **InstrokeAbscissaType** are **expected** to be mutually consistent with **Catch**, **Finish**, and drive timing when those scalars exist. **rowingdata does not validate** that relationship; consumers may perform their own checks if strict consistency is required.

**Parameters:** `instroke_abscissa_type=None` selects type **1** when stroke drive time exists, else **0**. `instroke_sample_interval_ms` overrides the per-stroke interval array (scalar broadcast or one value per stroke).

**Companion JSON** (see below) includes a **`_rowingdata_instroke`** object with `version`, `instroke_abscissa_type`, `instroke_point_count`, and `instroke_sample_interval_ms` (per-stroke list, milliseconds when type is time-based).

Non-uniform abscissas are not stored in FIT (255-byte field limit); resample to uniform spacing or use the companion file and optional future `x` arrays in JSON.

## In-stroke curve export

When `instroke_export` is not `'off'`, comma-separated curve columns (RP3 `curve_data`, Quiske `boat accelerator curve`, `oar angle velocity curve`, `seat curve`) can be exported:

| instroke_export | Behavior |
|-----------------|----------|
| `'off'` (default) | No curve export; backward compatible. |
| `'summary'` | Per-stroke metrics (q1, q2, q3, q4, diff, maxpos, minpos) as developer fields, e.g. `HandleForceCurve_q1`. |
| `'downsampled'` | Fixed-length curve per stroke as SINT16 array. Use `instroke_downsample_points` (default 16, range 2–127). |
| `'full'` | Full-resolution curve up to 127 points per stroke. Curves with ≤127 points are stored as-is (padded if shorter); longer curves are downsampled to 127. FIT developer fields have a 255-byte limit, so SINT16 (2 bytes) allows max 127 points. |
| `'companion'` | Sidecar `.instroke.json` file with full curve data per stroke (no FIT size limit). |

**Column mapping**: `curve_data` → HandleForceCurve, `boat accelerator curve` → BoatAcceleratorCurve, `oar angle velocity curve` → OarAngleVelocityCurve, `seat curve` → SeatCurve. Override via `instroke_column_map`. Columns are auto-detected when `instroke_columns` is None.

**Companion file format:** JSON object whose keys are canonical curve names (e.g. `HandleForceCurve`) mapping to a list of strokes, each stroke a list of numeric samples. A **`_rowingdata_instroke`** key (if present) holds metadata: `version`, `instroke_abscissa_type`, `instroke_point_count`, and `instroke_sample_interval_ms` (per-stroke list). Existing consumers that only read curve arrays remain compatible if they ignore unknown keys.

**instroke_downsample_points**: For `'downsampled'` mode, the number of points per stroke (default 16). Valid range 2–127. For `'full'` mode this parameter is ignored.

### In-FIT curve size limit (255 bytes)

The FIT protocol encodes developer field size in one byte, so each field is limited to 255 bytes. For SINT16 arrays that yields at most 127 points. We support `'full'` mode (up to 127 points) and configurable `'downsampled'` (2–127 points). For longer curves or lossless storage, use `'companion'`.

### Alternative approaches (not implemented)

**Option C – Two-part split**: Curves with >127 points could be split into two developer fields (e.g. `HandleForceCurve_Part0`, `HandleForceCurve_Part1`), each ≤127 points, giving up to 254 points total. Consumers would need to concatenate parts. This adds complexity to both export and parsing.

**Option E – UINT8/BYTE**: Using 1 byte per value would allow up to 255 points in a single field, but with lower precision (8-bit vs 16-bit). Scale/offset would map float curves to 0–255. Not implemented due to precision loss.

## Ecosystem and field stability

Field names and enums in this document are the **rowingdata** convention for FIT developer data (`application_id` `rowingdata`). Downstream tools (e.g. [Intervals.icu](https://intervals.icu)) can import these when they read developer field descriptions. If you maintain a consumer, coordinate renames or enum additions with this repo or file an issue before relying on new IDs in production.

## Missing columns

If a column is missing, the corresponding FIT field is omitted or zeroed. cum_dist is computed from Horizontal (meters) when absent. Developer fields are only written when the column exists.
