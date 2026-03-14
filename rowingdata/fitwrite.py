"""
FIT file export for rowingdata.
Exports rowingdata DataFrames to Garmin FIT format for compatibility with
Intervals.icu and other platforms.
"""
from __future__ import absolute_import
from __future__ import print_function

import datetime
import numpy as np
from dateutil import parser as ps
import arrow

try:
    from fit_tool.base_type import BaseType
    from fit_tool.developer_field import DeveloperField
    from fit_tool.fit_file_builder import FitFileBuilder
    from fit_tool.profile.messages.developer_data_id_message import DeveloperDataIdMessage
    from fit_tool.profile.messages.field_description_message import FieldDescriptionMessage
    from fit_tool.profile.messages.file_id_message import FileIdMessage
    from fit_tool.profile.messages.activity_message import ActivityMessage
    from fit_tool.profile.messages.session_message import SessionMessage
    from fit_tool.profile.messages.lap_message import LapMessage
    from fit_tool.profile.messages.record_message import RecordMessage
    from fit_tool.profile.messages.event_message import EventMessage
    from fit_tool.profile.profile_type import (
        FileType, Manufacturer, Sport, SubSport,
        Event, EventType, Activity,
    )
    FIT_TOOL_AVAILABLE = True
except ImportError:
    FIT_TOOL_AVAILABLE = False

# Developer field definitions for rowing-specific columns (no native FIT equivalent).
# Per README spec: DriveLength = handle distance (projection on longitudinal axis);
# StrokeDistance = distance traveled during stroke cycle (boat/erg travel).
# Native cycle_length is uint8/scale100 (max 2.55m), wrong for rowing (7–12m typical).
# (field_id, df_column, name, base_type, size, scale, units)
ROWING_DEV_FIELDS = [
    (0, ' DriveLength (meters)', 'DriveLength', BaseType.UINT16, 2, 100, 'm'),
    (1, ' DriveTime (ms)', 'StrokeDriveTime', BaseType.UINT16, 2, 1, 'ms'),
    (2, ' DragFactor', 'DragFactor', BaseType.UINT16, 2, 1, ''),
    (3, ' StrokeRecoveryTime (ms)', 'StrokeRecoveryTime', BaseType.UINT16, 2, 1, 'ms'),
    (4, ' AverageDriveForce (lbs)', 'AverageDriveForceLbs', BaseType.UINT16, 2, 10, 'lbs'),
    (5, ' PeakDriveForce (lbs)', 'PeakDriveForceLbs', BaseType.UINT16, 2, 10, 'lbs'),
    (6, ' AverageDriveForce (N)', 'AverageDriveForceN', BaseType.UINT16, 2, 10, 'N'),
    (7, ' PeakDriveForce (N)', 'PeakDriveForceN', BaseType.UINT16, 2, 10, 'N'),
    (8, ' AverageBoatSpeed (m/s)', 'AverageBoatSpeed', BaseType.UINT16, 2, 100, 'm/s'),
    (9, ' WorkoutState', 'WorkoutState', BaseType.UINT8, 1, 1, ''),
    (10, ' StrokeDistance (meters)', 'StrokeDistance', BaseType.UINT16, 2, 100, 'm'),
]

# Sport string to FIT enum mapping
SPORT_MAP = {
    'rowing': Sport.ROWING,
    'water': Sport.ROWING,
    'indoor_rowing': Sport.ROWING,
    'indoor rowing': Sport.ROWING,
    'run': Sport.RUNNING,
    'running': Sport.RUNNING,
    'bike': Sport.CYCLING,
    'cycling': Sport.CYCLING,
    'swim': Sport.SWIMMING,
    'swimming': Sport.SWIMMING,
    'other': Sport.GENERIC,
}


def _valid_position(lat, lon):
    """True if lat/lon are valid degrees for FIT (rough bounds)."""
    return (not (np.isnan(lat) or np.isnan(lon)) and
            -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0))


def _sport_to_fit(sport_str):
    """Map sport string to FIT Sport enum."""
    if sport_str is None:
        return Sport.ROWING
    key = str(sport_str).lower().strip()
    return SPORT_MAP.get(key, Sport.ROWING)


# Work stroke WorkoutState values (1,4,5,6,7,8,9 = work; 3 = rest per Garmin/rowing convention)
WORKOUT_STATES_WORK = [1, 4, 5, 6, 7, 8, 9]


def _compute_interval_summaries(df, lap_col, unixtimes, distance_m, heart_rate, cadence,
                                power, work_mask=None):
    """
    Compute per-interval summary stats for Lap messages.
    Returns list of dicts with keys: start_time_ms, total_elapsed_s, total_distance,
    total_calories, avg_heart_rate, max_heart_rate, avg_cadence, avg_power, indices.
    Uses work strokes only for avg HR, cadence, power (matches intervalstats).
    """
    try:
        calories_arr = df[' Calories (kCal)'].values
    except KeyError:
        calories_arr = None

    if work_mask is None:
        work_mask = np.ones(len(df), dtype=bool)

    summaries = []
    prev_max_dist = 0.0
    prev_max_cal = 0.0

    # Preserve order of first appearance (chronological)
    _, idx = np.unique(df[lap_col].values, return_index=True)
    interval_nrs = df[lap_col].values[np.sort(idx)]

    for lap_val in interval_nrs:
        mask = (df[lap_col].values == lap_val)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        start_time_ms = int(unixtimes[indices[0]] * 1000)
        end_time_ms = int(unixtimes[indices[-1]] * 1000)
        total_elapsed_s = (unixtimes[indices[-1]] - unixtimes[indices[0]]) if len(indices) > 1 else 0.0

        interval_dist = float(distance_m[indices[-1]]) - prev_max_dist
        if interval_dist < 0:
            interval_dist = 0
        prev_max_dist = float(distance_m[indices[-1]])

        total_cal = 0
        if calories_arr is not None:
            cal_max = np.nanmax(calories_arr[indices])
            total_cal = max(0, int(cal_max - prev_max_cal))
            prev_max_cal = cal_max

        work_in_interval = mask & work_mask
        work_idx = np.where(work_in_interval)[0]

        hr_vals = heart_rate[work_idx]
        hr_vals = hr_vals[hr_vals > 0]
        avg_hr = int(np.mean(hr_vals)) if len(hr_vals) > 0 else 0
        max_hr = int(np.max(heart_rate[indices])) if len(indices) > 0 else 0

        cad_vals = cadence[work_idx]
        cad_vals = cad_vals[cad_vals > 0]
        avg_cad = int(np.mean(cad_vals)) if len(cad_vals) > 0 else 0

        pw_vals = power[work_idx]
        pw_vals = pw_vals[pw_vals > 0]
        avg_pw = int(np.mean(pw_vals)) if len(pw_vals) > 0 else 0

        summaries.append({
            'start_time_ms': start_time_ms,
            'total_elapsed_s': total_elapsed_s,
            'total_distance': interval_dist,
            'total_calories': total_cal,
            'avg_heart_rate': avg_hr,
            'max_heart_rate': max_hr,
            'avg_cadence': avg_cad,
            'avg_power': avg_pw,
            'indices': indices,
        })

    return summaries


def write_fit(file_name, df, row_date="2016-01-01", notes="Exported by Rowingdata",
              sport="rowing", use_developer_fields=True):
    """
    Write rowingdata DataFrame to a FIT activity file.

    Parameters
    ----------
    file_name : str
        Output file path (e.g. 'activity.fit')
    df : pandas.DataFrame
        DataFrame with rowingdata columns (TimeStamp, Horizontal, Cadence, etc.)
    row_date : str or datetime
        Workout date (used when timestamps are relative)
    notes : str
        Activity notes
    sport : str
        Sport type: 'rowing', 'indoor_rowing', 'other', etc.
    use_developer_fields : bool
        If True, include rowing-specific columns as developer fields when present.
        If False, export only standard FIT fields (timestamp, distance, cadence,
        heart_rate, power, speed, position).
    """
    if not FIT_TOOL_AVAILABLE:
        raise ImportError("fit-tool is required for FIT export. Install with: pip install fit-tool")

    # Get or compute cum_dist
    if 'cum_dist' not in df.columns and ' Horizontal (meters)' in df.columns:
        from .csvparsers import make_cumvalues
        res = make_cumvalues(df[' Horizontal (meters)'])
        df = df.copy()
        df['cum_dist'] = res[0]

    nr_rows = len(df)
    if nr_rows == 0:
        raise ValueError("Cannot export empty DataFrame to FIT")

    # Parse row_date and determine timestamps
    dateobj = ps.parse(str(row_date))
    timezero = arrow.get(datetime.datetime(2000, 1, 1)).timestamp()
    seconds = df['TimeStamp (sec)'].values

    if seconds[0] < timezero:
        # Relative timestamps: add row_date
        unixtimes = seconds + arrow.get(dateobj).timestamp()
    else:
        unixtimes = seconds

    # FIT timestamps in milliseconds (fit-tool convention for timestamp fields)
    start_time_ms = int(unixtimes[0] * 1000)
    total_elapsed_ms = int((unixtimes[-1] - unixtimes[0]) * 1000) if nr_rows > 1 else 0
    total_elapsed_s = (unixtimes[-1] - unixtimes[0]) if nr_rows > 1 else 0.0  # seconds, for Activity/Session/Lap

    # Arrays for record messages
    try:
        distance_m = df['cum_dist'].values
    except KeyError:
        distance_m = df[' Horizontal (meters)'].values

    try:
        cadence = np.round(df[' Cadence (stokes/min)'].values).astype(int)
    except KeyError:
        cadence = np.zeros(nr_rows, dtype=int)

    try:
        heart_rate = df[' HRCur (bpm)'].values.astype(int)
        heart_rate = np.clip(heart_rate, 0, 255)  # uint8
    except KeyError:
        heart_rate = np.zeros(nr_rows, dtype=int)

    try:
        power = df[' Power (watts)'].values.astype(int)
        power = np.clip(power, 0, 65535)  # uint16
    except KeyError:
        power = np.zeros(nr_rows, dtype=int)

    try:
        pace = df[' Stroke500mPace (sec/500m)'].values
        # enhanced_speed in m/s; avoid div by zero
        pace_safe = np.where(pace > 0, pace, np.inf)
        enhanced_speed = 500.0 / pace_safe
        enhanced_speed = np.where(np.isfinite(enhanced_speed), enhanced_speed, 0)
    except KeyError:
        enhanced_speed = np.zeros(nr_rows)

    try:
        lat = df[' latitude'].values
        lon = df[' longitude'].values
    except KeyError:
        lat = np.zeros(nr_rows)
        lon = np.zeros(nr_rows)

    # Stroke number for native total_cycles field (data is per-stroke, one record per stroke)
    try:
        stroke_number = df[' Stroke Number'].values.astype(int)
    except KeyError:
        try:
            stroke_number = df['sessionStrokeIndex'].values.astype(int) + 1  # 1-based
        except KeyError:
            stroke_number = np.arange(1, nr_rows + 1, dtype=int)  # 1-based row index

    # Developer fields: which columns exist and their arrays
    use_dev = use_developer_fields and FIT_TOOL_AVAILABLE
    dev_arrays = {}
    dev_specs = []
    DEV_DATA_IDX = 0
    if use_dev:
        for fd in ROWING_DEV_FIELDS:
            field_id, col, name, base_type, size, scale, units = fd
            if col in df.columns:
                arr = df[col].values
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                # Clip to avoid overflow: encoded = value * scale must fit in base_type
                if base_type == BaseType.UINT8:
                    arr = np.clip(arr, 0, 255)
                elif base_type == BaseType.UINT16:
                    max_display = 65535.0 / scale if scale else 65535.0
                    arr = np.clip(arr, 0, max_display)
                dev_arrays[field_id] = arr
                dev_specs.append((field_id, col, name, base_type, size, scale, units))

    # Build FIT file
    min_str = 50 if dev_specs else 8
    builder = FitFileBuilder(auto_define=True, min_string_size=min_str)

    # File ID
    file_id = FileIdMessage()
    file_id.type = FileType.ACTIVITY
    file_id.manufacturer = Manufacturer.GARMIN.value
    file_id.product = 0
    file_id.time_created = start_time_ms
    file_id.serial_number = 0x12345678
    builder.add(file_id)

    # Activity message
    activity = ActivityMessage()
    activity.timestamp = start_time_ms
    activity.total_timer_time = total_elapsed_s  # seconds; fit-tool applies scale 1000
    activity.num_sessions = 1
    activity.type = Activity.MANUAL
    activity.event = Event.TIMER
    activity.event_type = EventType.START
    builder.add(activity)

    # Timer start event
    event_start = EventMessage()
    event_start.event = Event.TIMER
    event_start.event_type = EventType.START
    event_start.timestamp = start_time_ms
    builder.add(event_start)

    # Session message
    total_dist = float(distance_m[-1]) if nr_rows > 0 else 0
    try:
        total_calories = int(df[' Calories (kCal)'].max())
    except (KeyError, ValueError):
        total_calories = 0

    avg_hr = int(np.mean(heart_rate[heart_rate > 0])) if np.any(heart_rate > 0) else 0
    max_hr = int(np.max(heart_rate)) if nr_rows > 0 else 0
    avg_cadence = int(np.mean(cadence[cadence > 0])) if np.any(cadence > 0) else 0
    avg_power = int(np.mean(power[power > 0])) if np.any(power > 0) else 0

    session = SessionMessage()
    session.message_index = 0
    session.timestamp = start_time_ms
    session.start_time = start_time_ms
    session.total_elapsed_time = total_elapsed_s  # seconds; fit-tool applies scale 1000
    session.total_timer_time = total_elapsed_s
    session.total_distance = int(round(total_dist))  # fit-tool applies scale 100
    session.total_calories = total_calories
    session.sport = _sport_to_fit(sport)
    session.sub_sport = SubSport.GENERIC
    if avg_hr > 0:
        session.avg_heart_rate = avg_hr
    if max_hr > 0:
        session.max_heart_rate = max_hr
    if avg_cadence > 0:
        session.avg_cadence = avg_cadence
    if avg_power > 0:
        session.avg_power = avg_power
    builder.add(session)

    # Developer data (ID + field descriptions) when we have developer fields
    DEV_DATA_IDX = 0
    if dev_specs:
        dev_id_msg = DeveloperDataIdMessage()
        dev_id_msg.application_id = b'rowingdata'
        dev_id_msg.developer_data_index = DEV_DATA_IDX
        builder.add(dev_id_msg)
        for field_id, col, name, base_type, size, scale, units in dev_specs:
            fd_msg = FieldDescriptionMessage()
            fd_msg.developer_data_index = DEV_DATA_IDX
            fd_msg.field_definition_number = field_id
            fd_msg.fit_base_type_id = base_type.value
            fd_msg.field_name = name
            fd_msg.scale = int(scale) if scale == int(scale) else scale
            fd_msg.offset = 0
            fd_msg.units = units
            builder.add(fd_msg)

    # Lap column: rowingdata uses ' lapIdx'; some CSVs use 'lapIdx'
    lap_col = ' lapIdx' if ' lapIdx' in df.columns else ('lapIdx' if 'lapIdx' in df.columns else None)
    work_mask = None
    if lap_col is not None:
        try:
            ws = df[' WorkoutState'].values if ' WorkoutState' in df.columns else df['WorkoutState'].values
            work_mask = np.isin(ws.astype(int), WORKOUT_STATES_WORK)
        except (KeyError, TypeError):
            work_mask = np.ones(nr_rows, dtype=bool)

    # Determine if we have multiple intervals (per-interval Lap messages)
    interval_summaries = None
    if lap_col is not None:
        unique_laps = np.unique(df[lap_col].values)
        if len(unique_laps) > 1:
            interval_summaries = _compute_interval_summaries(
                df, lap_col, unixtimes, distance_m, heart_rate, cadence, power, work_mask
            )

    def _emit_record(i):
        """Emit a single Record message for row index i."""
        dev_fields = []
        if dev_specs:
            for field_id, col, name, base_type, size, scale, units in dev_specs:
                arr = dev_arrays[field_id]
                val = float(arr[i])
                if val != 0 or field_id == 9:  # WorkoutState can be 0
                    dev = DeveloperField(
                        developer_data_index=DEV_DATA_IDX,
                        field_id=field_id,
                        size=size,
                        name=name,
                        base_type=base_type,
                        scale=scale,
                        offset=0,
                        units=units
                    )
                    dev.set_value(0, val)
                    dev_fields.append(dev)
        rec = RecordMessage(developer_fields=dev_fields) if dev_fields else RecordMessage()
        rec.timestamp = int(unixtimes[i] * 1000)
        rec.distance = float(distance_m[i])
        rec.heart_rate = heart_rate[i] if heart_rate[i] > 0 else None
        rec.cadence = cadence[i] if cadence[i] > 0 else None
        rec.power = power[i] if power[i] > 0 else None
        rec.enhanced_speed = float(enhanced_speed[i]) if enhanced_speed[i] > 0 else None
        if hasattr(rec, 'total_cycles'):
            rec.total_cycles = int(stroke_number[i])
        if not (np.isnan(lat[i]) or lat[i] == 0) and not (np.isnan(lon[i]) or lon[i] == 0):
            rec.position_lat = float(lat[i])
            rec.position_long = float(lon[i])
        builder.add(rec)

    if interval_summaries is not None and len(interval_summaries) > 0:
        # Multi-interval: Event(lap start), Lap, Records per interval
        for lap_idx, summ in enumerate(interval_summaries):
            ev = EventMessage()
            ev.event = Event.LAP
            ev.event_type = EventType.START
            ev.timestamp = summ['start_time_ms']
            builder.add(ev)

            lap = LapMessage()
            lap.message_index = lap_idx
            lap.timestamp = summ['start_time_ms']
            lap.start_time = summ['start_time_ms']
            lap.total_elapsed_time = summ['total_elapsed_s']
            lap.total_timer_time = summ['total_elapsed_s']
            lap.total_distance = int(round(summ['total_distance']))
            lap.total_calories = summ['total_calories']
            lap.sport = _sport_to_fit(sport)
            lap.sub_sport = SubSport.GENERIC
            if summ['avg_heart_rate'] > 0:
                lap.avg_heart_rate = summ['avg_heart_rate']
            if summ['max_heart_rate'] > 0:
                lap.max_heart_rate = summ['max_heart_rate']
            if summ['avg_cadence'] > 0:
                lap.avg_cadence = summ['avg_cadence']
            if summ['avg_power'] > 0:
                lap.avg_power = summ['avg_power']
            builder.add(lap)

            for i in summ['indices']:
                _emit_record(i)
    else:
        # Single lap: one Lap for whole session, then all Records
        lap = LapMessage()
        lap.message_index = 0
        lap.timestamp = start_time_ms
        lap.start_time = start_time_ms
        lap.total_elapsed_time = total_elapsed_s
        lap.total_timer_time = total_elapsed_s
        lap.total_distance = int(round(total_dist))
        lap.total_calories = total_calories
        lap.sport = _sport_to_fit(sport)
        lap.sub_sport = SubSport.GENERIC
        if avg_hr > 0:
            lap.avg_heart_rate = avg_hr
        if max_hr > 0:
            lap.max_heart_rate = max_hr
        if avg_cadence > 0:
            lap.avg_cadence = avg_cadence
        if avg_power > 0:
            lap.avg_power = avg_power
        builder.add(lap)

        for i in range(nr_rows):
            _emit_record(i)

    # Timer stop event
    event_stop = EventMessage()
    event_stop.event = Event.TIMER
    event_stop.event_type = EventType.STOP_ALL
    event_stop.timestamp = int(unixtimes[-1] * 1000) if nr_rows > 0 else start_time_ms
    builder.add(event_stop)

    fit_file = builder.build()
    fit_file.to_file(file_name)
