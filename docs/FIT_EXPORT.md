# FIT Export

The rowingdata library can export rowing sessions to Garmin FIT format for use with [Intervals.icu](https://intervals.icu) and other platforms that support FIT files.

## Usage

```python
import rowingdata

row = rowingdata.rowingdata(csvfile="workout.csv")
row.exporttofit("workout.fit", sport="rowing", notes="My workout")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fileName` | — | Output file path (e.g. `activity.fit`) |
| `notes` | `"Exported by Rowingdata"` | Activity notes |
| `sport` | `"rowing"` | Sport type (see [Supported sports](#supported-sports)) |
| `use_developer_fields` | `True` | Include rowing-specific developer fields when present |

## Native vs developer fields

FIT distinguishes between two kinds of fields:

**Native fields** (also called *profile fields*) are part of the official Garmin FIT SDK. They are defined in the standard profile (e.g. `timestamp`, `distance`, `heart_rate`, `position_lat`, `cycle_length`). Every FIT-capable app understands these; they map directly to built-in streams in platforms like Intervals.icu.

**Developer fields** are custom fields defined by a manufacturer or developer. They require extra Developer Data ID and Field Description messages in the FIT file to declare their meaning, units, and scale. Consumer apps may ignore them unless they explicitly support that developer's fields.

**Current export:** rowingdata exports **native fields** for all standard metrics, plus **developer fields** for rowing-specific columns when `use_developer_fields=True` (default) and the column is present in the data.

### Developer fields exported

When `use_developer_fields=True`, the following rowing-specific columns are written as developer fields (in Record messages) when the column exists in the data:

| rowingdata column | Developer field name | Base type | Scale | Units |
|-------------------|----------------------|-----------|-------|-------|
| ` DriveLength (meters)` | DriveLength | UINT16 | 1000 | m |
| ` DriveTime (ms)` | DriveTime | UINT16 | 1 | ms |
| ` DragFactor` | DragFactor | UINT16 | 1 |  |
| ` StrokeRecoveryTime (ms)` | StrokeRecoveryTime | UINT16 | 1 | ms |
| ` AverageDriveForce (lbs)` | AverageDriveForceLbs | UINT16 | 10 | lbs |
| ` PeakDriveForce (lbs)` | PeakDriveForceLbs | UINT16 | 10 | lbs |
| ` AverageDriveForce (N)` | AverageDriveForceN | UINT16 | 10 | N |
| ` PeakDriveForce (N)` | PeakDriveForceN | UINT16 | 10 | N |
| ` AverageBoatSpeed (m/s)` | AverageBoatSpeed | UINT16 | 1000 | m/s |
| ` WorkoutState` | WorkoutState | UINT8 | 1 |  |

These have no native FIT equivalents. Apps that support custom streams (e.g. Intervals.icu) may import them when the developer field descriptions are present in the FIT file.

## Dependencies

FIT export requires the [fit-tool](https://pypi.org/project/fit-tool/) package:

```bash
pip install fit-tool
```

## Field mapping

The following table describes how rowingdata columns are mapped to FIT messages and fields. All fields listed are **native** unless otherwise noted.

### Record messages (per-sample data)

| rowingdata column | FIT field | Type | Notes |
|-------------------|-----------|------|-------|
| `TimeStamp (sec)` | `timestamp` | Native | Seconds since 1970-01-01 UTC; relative timestamps are combined with `row_date` |
| `cum_dist` or ` Horizontal (meters)` | `distance` | Native | Cumulative distance in meters (FIT scale 100) |
| ` Cadence (stokes/min)` | `cadence` | Native | Stroke rate in strokes/min; omitted if zero |
| ` HRCur (bpm)` | `heart_rate` | Native | Clamped to 0–255 (FIT uint8) |
| ` Power (watts)` | `power` | Native | Clamped to 0–65535 (FIT uint16) |
| ` Stroke500mPace (sec/500m)` | `enhanced_speed` | Native | Converted to m/s via 500/pace; FIT scale 1000 |
| ` latitude` | `position_lat` | Native | Degrees; omitted if missing or invalid |
| ` longitude` | `position_long` | Native | Degrees; omitted if missing or invalid |

### Session and lap summary fields

All session and lap fields are **native**.

| Source | FIT message | Fields |
|--------|-------------|--------|
| Session totals | `SessionMessage` | `total_distance`, `total_calories`, `avg_heart_rate`, `max_heart_rate`, `avg_cadence`, `avg_power` |
| Lap boundaries | `LapMessage` | Same as session; currently one lap for the whole workout |
| Position | `LapMessage` | `start_position_lat/long`, `end_position_lat/long` (degrees) when valid |

### Position handling

- Latitude and longitude are expected in **degrees** (WGS84).
- Position values must be in the range lat ∈ [-90, 90], lon ∈ [-180, 180].
- Values at 0,0 or NaN are treated as invalid and omitted.
- FIT uses semicircles internally; the export passes degrees and the library performs conversion.

### Cycle length (stroke distance)

- FIT `cycle_length` uses scale 100 and a uint8 base type, so the maximum representable value is 2.55 m.
- Values above 2.55 m are capped; values above 10 m in the source are pre-clipped.

## Supported sports

| sport string | FIT Sport enum |
|-------------|----------------|
| `rowing`, `water`, `indoor_rowing`, `indoor rowing` | ROWING |
| `run`, `running` | RUNNING |
| `bike`, `cycling` | CYCLING |
| `swim`, `swimming` | SWIMMING |
| `other` | GENERIC |

## Timestamps

- If `TimeStamp (sec)` values are below the year-2000 epoch (e.g. relative seconds from workout start), they are combined with `row_date` from the rowingdata session.
- FIT timestamps are stored in milliseconds since 1989-12-31 00:00:00 UTC (Garmin epoch).

## FIT file structure

Exported files include:

1. **File ID** – Activity type, manufacturer (Garmin), creation time
2. **Activity** – Timer start metadata
3. **Event** – Timer start
4. **Session** – Summary (distance, calories, HR, cadence, power)
5. **Lap** – One lap spanning the full session (with optional start/end position)
6. **Record** – One record per row (timestamp, distance, cadence, HR, power, speed, position, cycle_length)
7. **Event** – Timer stop

## Missing or optional columns

- If a column is missing, the corresponding FIT field is omitted or set to zero.
- `cum_dist` is computed from ` Horizontal (meters)` when not present.
- When `use_developer_fields=True`, developer fields are included only for columns that exist in the data.
