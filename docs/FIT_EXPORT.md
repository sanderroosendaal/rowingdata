# FIT Export – rowingdata support for Intervals.icu

The rowingdata library exports rowing sessions to Garmin FIT format for use with [Intervals.icu](https://intervals.icu) and other platforms. Here's what we export and how it maps.

## Usage

```python
import rowingdata

row = rowingdata.rowingdata(csvfile="workout.csv")
row.exporttofit("workout.fit", sport="rowing", notes="My workout")
```

Parameters: **fileName** (output path), **notes** (default: "Exported by Rowingdata"), **sport** (e.g. rowing, indoor_rowing), **use_developer_fields** (default: True – include rowing-specific fields when present).

## Native vs developer fields

**Native fields** are part of the Garmin FIT SDK (timestamp, distance, heart_rate, position_lat, cycle_length, etc.). Every FIT-capable app understands them.

**Developer fields** are custom fields defined in the FIT file. They need extra metadata; apps may ignore them unless they explicitly support rowingdata's fields.

We export native fields for standard metrics plus developer fields for rowing-specific columns when use_developer_fields=True and the column exists.

## Naming and field choices

- **StrokeDriveTime** – drive phase time (ms). Named for symmetry with StrokeRecoveryTime.
- **DriveLength** – handle distance (meters). Per README: distance traveled by the handle along the longitudinal axis. For indoor rowing, handle travel catch-to-finish.
- **StrokeDistance** – distance traveled during the stroke cycle (meters). Per README. Developer field used because Garmin native cycle_length maxes at 2.55 m.
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

## Session and lap

Session totals (total_distance, total_calories, avg_heart_rate, max_heart_rate, avg_cadence, avg_power) and lap boundaries are native. Currently one lap for the whole workout. Start/end position in degrees when valid.

## Dependencies

```bash
pip install fit-tool
```

## Timestamps and structure

- Relative timestamps (below year 2000) are combined with row_date.
- FIT timestamps: milliseconds since Garmin epoch (1989-12-31 UTC).
- File structure: File ID, Activity, Event (start), Session, Lap, Record messages (one per row), Event (stop).

## Missing columns

If a column is missing, the corresponding FIT field is omitted or zeroed. cum_dist is computed from Horizontal (meters) when absent. Developer fields are only written when the column exists.
