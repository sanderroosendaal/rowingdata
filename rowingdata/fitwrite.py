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
# StrokeDistance is here: native cycle_length is uint8/scale100 (max 2.55m), wrong for rowing (7–12m typical).
# (field_id, df_column, name, base_type, size, scale, units)
ROWING_DEV_FIELDS = [
    (0, ' DriveLength (meters)', 'DriveLength', BaseType.UINT16, 2, 100, 'm'),
    (1, ' DriveTime (ms)', 'DriveTime', BaseType.UINT16, 2, 1, 'ms'),
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

    # FIT timestamps in milliseconds (fit-tool convention)
    start_time_ms = int(unixtimes[0] * 1000)
    total_elapsed_ms = int((unixtimes[-1] - unixtimes[0]) * 1000) if nr_rows > 1 else 0

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
    activity.total_timer_time = total_elapsed_ms
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
    session.total_elapsed_time = total_elapsed_ms
    session.total_timer_time = total_elapsed_ms
    session.total_distance = int(total_dist * 100)  # scale 100
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

    # Lap message (one lap for the whole session for now)
    lap = LapMessage()
    lap.message_index = 0
    lap.timestamp = int(unixtimes[-1] * 1000) if nr_rows > 0 else start_time_ms
    lap.start_time = start_time_ms
    lap.total_elapsed_time = total_elapsed_ms
    lap.total_timer_time = total_elapsed_ms
    lap.total_distance = int(total_dist * 100)
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
    # fit-tool expects position in degrees (it applies semicircle conversion internally)
    if _valid_position(lat[0], lon[0]) and _valid_position(lat[-1], lon[-1]):
        lap.start_position_lat = float(lat[0])
        lap.start_position_long = float(lon[0])
        lap.end_position_lat = float(lat[-1])
        lap.end_position_long = float(lon[-1])
    builder.add(lap)

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

    # Record messages
    for i in range(nr_rows):
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
        rec.distance = int(distance_m[i] * 100)  # scale 100
        rec.heart_rate = heart_rate[i] if heart_rate[i] > 0 else None
        rec.cadence = cadence[i] if cadence[i] > 0 else None
        rec.power = power[i] if power[i] > 0 else None
        rec.enhanced_speed = int(enhanced_speed[i] * 1000) if enhanced_speed[i] > 0 else None  # scale 1000

        if not (np.isnan(lat[i]) or lat[i] == 0) and not (np.isnan(lon[i]) or lon[i] == 0):
            rec.position_lat = float(lat[i])
            rec.position_long = float(lon[i])

        builder.add(rec)

    # Timer stop event
    event_stop = EventMessage()
    event_stop.event = Event.TIMER
    event_stop.event_type = EventType.STOP_ALL
    event_stop.timestamp = int(unixtimes[-1] * 1000) if nr_rows > 0 else start_time_ms
    builder.add(event_stop)

    fit_file = builder.build()
    fit_file.to_file(file_name)
