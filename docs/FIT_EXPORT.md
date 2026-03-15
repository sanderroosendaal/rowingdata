# FIT Export – rowingdata support for Intervals.icu

The rowingdata library exports rowing sessions to Garmin FIT format for use with [Intervals.icu](https://intervals.icu) and other platforms. Here's what we export and how it maps.

Per the [Garmin FIT SDK Activity structure](https://developer.garmin.com/fit/file-types/activity/), FIT files use Activity, Session, Lap, and Record messages. We emit summary data at session level (whole workout), split/interval level (per lapIdx), and record level (per stroke).

## Usage

```python
import rowingdata

row = rowingdata.rowingdata(csvfile="workout.csv")
row.exporttofit("workout.fit", sport="rowing", notes="My workout")
```

Parameters: **fileName** (output path), **notes** (default: "Exported by Rowingdata"), **sport** (e.g. rowing, indoor_rowing), **use_developer_fields** (default: True – include rowing-specific fields when present), **instroke_export** (default: 'off' – see In-stroke curve export below), **instroke_columns** (optional list of curve columns), **instroke_column_map** (optional override mapping), **instroke_downsample_points** (default: 16 for downsampled export).

## Native vs developer fields

**Native fields** are part of the Garmin FIT SDK (timestamp, distance, heart_rate, position_lat, cycle_length, etc.). Every FIT-capable app understands them.

**Developer fields** are custom fields defined in the FIT file. They need extra metadata; apps may ignore them unless they explicitly support rowingdata's fields.

We export native fields for standard metrics plus developer fields for rowing-specific columns when use_developer_fields=True and the column exists.

## Naming and field choices

- **StrokeDriveTime** – drive phase time (ms). Named for symmetry with StrokeRecoveryTime.
- **DriveLength** – handle distance (meters). Per README: distance traveled by the handle along the longitudinal axis of the boat or erg. For OTW rowing, this is the projection of the handle trajectory on the longitudinal axis (not the full path the hands travel). For indoor rowing, typically handle travel catch-to-finish. Typical range: 1.2–1.5 m. This is a *stroke output* metric.
- **StrokeDistance** – distance traveled during the stroke cycle (meters). Per README: the distance the boat/erg travels during one stroke cycle. Distinct from DriveLength (handle distance). Typical range: 7–12 m for OTW. Developer field used because Garmin native cycle_length maxes at 2.55 m, too small for rowing.
- **DriveLength vs EffectiveLength** – Not duplicates. **DriveLength** is the actual distance the handle traveled during the stroke (~1.2–1.5 m). **EffectiveLength** is an oarlock/rigging metric: the effective lever length (horizontal component from pin to handle, often in cm). It describes rigging geometry, varies slightly stroke-to-stroke, and comes from NK Logbook (Oarlock). Both can appear in the same dataset.
- **Stroke Number** – stored in the native **total_cycles** field (Garmin/ANT+ cycles). Rowing machines report it via ANT+ to Garmin watches; we write it per record since data is per stroke.

## Developer fields exported

| rowingdata column | FIT field name | Base type | Scale | Units |
|-------------------|----------------|-----------|-------|-------|
| DriveLength (meters) | DriveLength | UINT16 | 100 | m |
| DriveTime (ms) | StrokeDriveTime | UINT16 | 1 | ms |
| DragFactor | DragFactor | UINT16 | 1 |  |
| StrokeRecoveryTime (ms) | StrokeRecoveryTime | UINT16 | 1 | ms |
| AverageDriveForce (lbs) | AverageDriveForceLbs | UINT16 | 10 | lbs |
| PeakDriveForce (lbs) | PeakDriveForceLbs | UINT16 | 10 | lbs |
| AverageDriveForce (N) | AverageDriveForceN | UINT16 | 10 | N |
| PeakDriveForce (N) | PeakDriveForceN | UINT16 | 10 | N |
| AverageBoatSpeed (m/s) | AverageBoatSpeed | UINT16 | 100 | m/s |
| WorkoutState | WorkoutState | UINT8 | 1 |  |
| StrokeDistance (meters) | StrokeDistance | UINT16 | 100 | m |
| catch, catchAngle | Catch | SINT16 | 10 | deg |
| finish, finishAngle | Finish | SINT16 | 10 | deg |
| slip | Slip | SINT16 | 10 | deg |
| wash | Wash | SINT16 | 10 | deg |
| peakforceangle | PeakForceAngle | SINT16 | 10 | deg |
| effectiveLength | EffectiveLength | UINT16 | 100 | m |

Oarlock scalars (catch, finish, slip, wash, peakforceangle, effectiveLength) are exported when present. NK Logbook (Oarlock) uses these columns. See README *Oarlock scalars (OTW rigging)* for definitions. **EffectiveLength** is distinct from **DriveLength**: the former is rigging geometry (effective lever length); the latter is actual handle travel distance.

These have no native equivalents. Apps like Intervals.icu can import them when the developer field descriptions are present.

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
| Stroke Number or row index | total_cycles | Stroke number (data is per-stroke) |

## Session, Lap, and Event messages

- **Session** – One message for the whole workout (total_distance, total_calories, avg_heart_rate, max_heart_rate, avg_cadence, avg_power). Start/end position in degrees when valid.
- **Lap** – One Lap message per interval when the data has multiple unique `lapIdx` values (supports both ` lapIdx` and `lapIdx` column names). Each Lap has per-interval distance, elapsed time, total_calories, avg HR, max HR, avg cadence, avg power. Per-interval avg HR, cadence, and power use **work strokes only** (WorkoutState 1,4,5,6,7,8,9); rest strokes (WorkoutState 3) are excluded from those averages. If `lapIdx` is missing or all values are the same, one Lap for the whole session.
- **Event** – Lap boundaries are marked with Event messages (Event.LAP, EventType.START) before each lap's records. Timer start/stop events bracket the activity.

## Dependencies

```bash
pip install fit-tool
```

## Timestamps and structure

- Relative timestamps (below year 2000) are combined with row_date.
- FIT timestamps: milliseconds since Garmin epoch (1989-12-31 UTC).
- File structure: File ID, Activity, Event (start), Session, Developer data (if any), then for each interval: Event (lap start), Lap, Record messages for that interval, then Event (stop). If there is only one interval: one Lap, then all Records.

## Intervals.icu compatibility

We structure the FIT file for compatibility with [Intervals.icu](https://intervals.icu):

- **Lap-first ordering per interval**: Each Lap message immediately precedes the Record messages it summarizes (Summary First style per interval). Lap `timestamp` and `start_time` both use the lap start time.
- **No Split messages**: We do not emit Garmin Split messages. Garmin's Split and Lap messages are disconnected in FIT; parsers can misalign them. Using only Lap messages avoids this parsing issue.
- **sub_sport**: For rowing activities we infer from data when possible:
  - If `sport` is explicitly `indoor_rowing` or `indoor rowing` → `indoorRowing` (indoor/erg)
  - If `sport` is explicitly `water` → generic (on-water)
  - If `sport` is `rowing` or default → GPS present → generic (on-water OTW); no GPS → `indoorRowing`

## In-stroke curve export

When `instroke_export` is not `'off'`, comma-separated curve columns (RP3 `curve_data`, Quiske `boat accelerator curve`, `oar angle velocity curve`, `seat curve`) can be exported:

| instroke_export | Behavior |
|-----------------|----------|
| `'off'` (default) | No curve export; backward compatible. |
| `'summary'` | Per-stroke metrics (q1, q2, q3, q4, diff, maxpos, minpos) as developer fields, e.g. `HandleForceCurve_q1`. |
| `'downsampled'` | Fixed-length curve (default 16 points) per stroke as SINT16 array developer field. |
| `'companion'` | Sidecar `.instroke.json` file with full curve data per stroke. |

**Column mapping**: `curve_data` → HandleForceCurve, `boat accelerator curve` → BoatAcceleratorCurve, `oar angle velocity curve` → OarAngleVelocityCurve, `seat curve` → SeatCurve. Override via `instroke_column_map`. Columns are auto-detected when `instroke_columns` is None. Use `instroke_downsample_points` to set the number of points for downsampled export (default 16).

## Missing columns

If a column is missing, the corresponding FIT field is omitted or zeroed. cum_dist is computed from Horizontal (meters) when absent. Developer fields are only written when the column exists.
