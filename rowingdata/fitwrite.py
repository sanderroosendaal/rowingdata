"""
FIT file export for rowingdata.
Exports rowingdata DataFrames to Garmin FIT format for compatibility with
Intervals.icu and other platforms.
"""
from __future__ import absolute_import
from __future__ import print_function

import datetime
import json
import os
import numpy as np
import pandas as pd
from dateutil import parser as ps
import arrow

from . import fitwrite_spec
from . import fit_garmin_bridge

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

# FIT developer field definitions: authoritative list in rowingdata/data/fit_export_spec.json
_FIT_EXPORT_RAW = fitwrite_spec.load_fit_spec_raw()
_ae = _FIT_EXPORT_RAW['abscissa_enum']
INSTROKE_ABSCISSA_UNKNOWN = _ae['UNKNOWN']
INSTROKE_ABSCISSA_TIME_UNIFORM_MS = _ae['TIME_UNIFORM_MS']
INSTROKE_ABSCISSA_HANDLE_DISTANCE_UNIFORM_M = _ae['HANDLE_DISTANCE_UNIFORM_M']
INSTROKE_ABSCISSA_OAR_ANGLE_UNIFORM_DEG = _ae['OAR_ANGLE_UNIFORM_DEG']
INSTROKE_ABSCISSA_NORMALIZED_DRIVE_0_1 = _ae['NORMALIZED_DRIVE_0_1']

INSTROKE_AXIS_FIELD_IDS = tuple(_FIT_EXPORT_RAW['instroke_axis_field_ids'])

# FIT developer field size limit: 255 bytes. SINT16 = 2 bytes => max 127 points.
INSTROKE_MAX_POINTS = 127

# Developer fields that must be emitted even when value is 0 (semantic zero / unknown).
ALWAYS_EMIT_DEV_FIELD_IDS = frozenset(_FIT_EXPORT_RAW['always_emit_field_ids'])

# Canonical mapping: df column name -> FIT curve type name (RP3/Quiske); from fit_export_spec.json
INSTROKE_COLUMN_MAP = dict(_FIT_EXPORT_RAW['instroke_column_map'])

def _parse_instroke_curve(df, col):
    """Parse curve column to DataFrame of numeric values. Same format as rowingdata get_instroke_data."""
    d = df[col].astype(str).str[1:-1].str.split(',', expand=True)
    return d.apply(pd.to_numeric, errors='coerce')


def _compute_instroke_summary(df, col):
    """
    Compute per-stroke summary metrics for a curve column (q1..q4, diff, maxpos, minpos).
    Mirrors add_instroke_metrics, add_instroke_diff, add_instroke_maxminpos from rowingdata.
    Returns dict with keys q1,q2,q3,q4,diff,maxpos,minpos (each a 1D array of length len(df)).
    """
    curve = _parse_instroke_curve(df, col)
    dfnorm = curve.copy().abs()
    row_max = dfnorm.T.max()
    row_max = row_max.replace(0, np.nan)
    dfnorm = dfnorm.T / row_max.values
    dfnorm = dfnorm.T
    dfnorm = dfnorm.fillna(0)

    ncol = len(dfnorm.columns)
    markers = (np.arange(5) * ncol / 4).astype(int)
    q1 = dfnorm.iloc[:, markers[0]:markers[1]].mean(axis=1).rolling(10, min_periods=1).std().fillna(0).values
    q2 = dfnorm.iloc[:, markers[1]:markers[2]].mean(axis=1).rolling(10, min_periods=1).std().fillna(0).values
    q3 = dfnorm.iloc[:, markers[2]:markers[3]].mean(axis=1).rolling(10, min_periods=1).std().fillna(0).values
    q4 = dfnorm.iloc[:, markers[3]:markers[4]].mean(axis=1).rolling(10, min_periods=1).std().fillna(0).values

    diff_mat = dfnorm.diff(axis=0).fillna(0) ** 2
    diff_arr = (diff_mat.sum(axis=1) / float(ncol)).fillna(0).values

    min_idxs = dfnorm.idxmin(axis=1)
    max_idxs = dfnorm.idxmax(axis=1)
    maxpos = (max_idxs.astype(float) / ncol).fillna(0).values
    minpos = (min_idxs.astype(float) / ncol).fillna(0).values

    return {'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'diff': diff_arr, 'maxpos': maxpos, 'minpos': minpos}


def _downsample_instroke_curve(df, col, n_points):
    """Downsample each row's curve to n_points. Returns list of arrays, one per row."""
    curve = _parse_instroke_curve(df, col)
    result = []
    for i in range(len(curve)):
        row = curve.iloc[i].dropna().values
        row = row[np.isfinite(row)]
        if len(row) < 2:
            result.append(np.zeros(n_points))
            continue
        if len(row) <= n_points:
            padded = np.zeros(n_points)
            padded[:len(row)] = row
            result.append(padded)
            continue
        indices = np.linspace(0, len(row) - 1, n_points, dtype=int)
        result.append(row[indices].astype(float))
    return result


def _get_instroke_curve_for_export(df, col, mode, downsample_points):
    """
    Get in-stroke curve array for FIT export.
    mode: 'downsampled' or 'full'
    downsample_points: for 'downsampled', the desired number of points; ignored for 'full'.

    Returns (list of arrays per row, n_points). n_points is in range 2..INSTROKE_MAX_POINTS.
    """
    curve = _parse_instroke_curve(df, col)
    if curve.empty or len(curve) == 0:
        return [], 0

    if mode == 'full':
        max_in_col = 0
        for i in range(len(curve)):
            row = curve.iloc[i].dropna().values
            row = row[np.isfinite(row)]
            max_in_col = max(max_in_col, len(row))
        n_points = min(max(2, max_in_col), INSTROKE_MAX_POINTS)
    else:
        n_points = max(2, min(downsample_points, INSTROKE_MAX_POINTS))

    result = _downsample_instroke_curve(df, col, n_points)
    return result, n_points


def _detect_instroke_columns(df):
    """
    Detect columns containing comma-separated numeric curve data (in-stroke).
    Returns list of column names where some row parses to at least 2 numeric samples.

    We scan multiple leading rows (not only row 0). Garmin/ORM FIT files often omit
    HandleForceCurve on the first records; only checking row 0 would miss ``curve_data``.
    """
    cols = []
    max_scan = min(len(df), 500)
    for c in df.columns:
        try:
            ser = df[c]
            found = False
            for i in range(max_scan):
                raw = ser.iloc[i]
                if pd.isna(raw):
                    continue
                st = str(raw).strip()
                if st in ('', 'nan', 'None'):
                    continue
                if len(st) >= 2 and st[0] == '(' and st[-1] == ')':
                    inner = st[1:-1]
                else:
                    inner = st
                parts = [p.strip() for p in inner.split(',') if p.strip()]
                if len(parts) < 2:
                    continue
                numeric_count = 0
                for v in parts[:32]:
                    try:
                        x = pd.to_numeric(v, errors='coerce')
                        if pd.notna(x) and not (isinstance(x, float) and np.isnan(x)):
                            numeric_count += 1
                    except (TypeError, ValueError):
                        pass
                if numeric_count >= 2:
                    found = True
                    break
            if found:
                cols.append(c)
        except (IndexError, KeyError, AttributeError, TypeError, ValueError):
            pass
    return cols


def _series_to_peak_force_position_norm(values):
    """Map source columns to UINT16 0-10000 (ten-thousandths of unity) for PeakForcePositionNorm."""
    v = pd.to_numeric(values, errors='coerce').values
    out = np.zeros(len(v), dtype=np.float64)
    for i, x in enumerate(v):
        if not np.isfinite(x):
            continue
        if x > 1.5:
            frac = min(1.0, x / 100.0)
        else:
            frac = min(1.0, max(0.0, float(x)))
        out[i] = round(frac * 10000.0)
    return out


def _series_to_peak_force_position_abs_m(values):
    """Map peak_force_pos (often RP3 cm) to meters for PeakForcePositionAbs (scale 100)."""
    v = pd.to_numeric(values, errors='coerce').values
    out = np.zeros(len(v), dtype=np.float64)
    for i, x in enumerate(v):
        if not np.isfinite(x) or x <= 0:
            continue
        if x > 2.5:
            m = x / 100.0
        else:
            m = float(x)
        out[i] = min(m, 3.0)
    return out


def _compute_instroke_axis_arrays(df, n_points, instroke_abscissa_type, instroke_sample_interval_ms):
    """
    Per-record arrays for InstrokeAbscissaType (90), InstrokeSampleInterval (91), InstrokePointCount (92).
    Returns None if n_points < 2.
    """
    nr = len(df)
    if n_points < 2:
        return None
    if instroke_abscissa_type is not None:
        atype = int(instroke_abscissa_type)
    else:
        atype = (INSTROKE_ABSCISSA_TIME_UNIFORM_MS if ' DriveTime (ms)' in df.columns
                 else INSTROKE_ABSCISSA_UNKNOWN)
    if instroke_sample_interval_ms is not None:
        if np.isscalar(instroke_sample_interval_ms):
            interval = np.full(nr, float(instroke_sample_interval_ms))
        else:
            arr = np.asarray(instroke_sample_interval_ms, dtype=np.float64).reshape(-1)
            interval = arr if len(arr) == nr else np.zeros(nr)
    elif atype == INSTROKE_ABSCISSA_TIME_UNIFORM_MS and ' DriveTime (ms)' in df.columns:
        drive_ms = np.nan_to_num(df[' DriveTime (ms)'].values, nan=0.0, posinf=0.0, neginf=0.0)
        interval = np.round(drive_ms / np.maximum(n_points - 1, 1))
        interval = np.clip(interval, 0, 65535)
    else:
        interval = np.zeros(nr)
    pc = np.full(nr, float(min(int(n_points), 127)))
    ta = np.full(nr, float(atype))
    return ta, interval, pc


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


def _sub_sport_for_sport(sport_str, has_gps=None):
    """
    Return sub_sport for rowing activities.
    - Explicit 'indoor_rowing'/'indoor rowing' -> INDOOR_ROWING
    - Explicit 'water' -> GENERIC (on-water)
    - 'rowing' or default -> infer from has_gps: GPS present -> GENERIC (OTW), else INDOOR_ROWING
    - Non-rowing sports -> GENERIC
    """
    if sport_str is None:
        key = 'rowing'
    else:
        key = str(sport_str).lower().strip()
    if key not in ('rowing', 'water', 'indoor_rowing', 'indoor rowing'):
        return SubSport.GENERIC
    if key in ('indoor_rowing', 'indoor rowing'):
        return SubSport.INDOOR_ROWING
    if key == 'water':
        return SubSport.GENERIC  # on-water
    # key == 'rowing' (default): infer from GPS
    if has_gps:
        return SubSport.GENERIC  # on-water (OTW)
    return SubSport.INDOOR_ROWING  # indoor/erg


# Work stroke WorkoutState values (1,4,5,6,7,8,9 = work; 3 = rest per Garmin/rowing convention)
WORKOUT_STATES_WORK = [1, 4, 5, 6, 7, 8, 9]

# Minimum Lap total_elapsed_time / total_timer_time (seconds). FIT uses ms scale in file; true
# zero is rejected by some viewers when a lap has only one Record (first-to-last stroke span is 0).
MIN_FIT_LAP_ELAPSED_S = 1e-3


def _compute_interval_summaries(df, lap_col, unixtimes, distance_m, heart_rate, cadence,
                                power, work_mask=None):
    """
    Compute per-interval summary stats for Lap messages.
    Returns list of dicts with keys: start_time_ms, total_elapsed_s, total_distance,
    total_calories, avg_heart_rate, max_heart_rate, avg_cadence, avg_power, indices.
    Uses work strokes only for avg HR, cadence, power (matches intervalstats).

    Per Garmin FIT semantics, Lap **total_elapsed_time** / **total_timer_time** use **wall-clock**
    duration from the **first** stroke of the lap to the **first** stroke of the **next** lap
    (or last stroke of the session for the final lap), not only last-minus-first stroke *within*
    the lap. Single-stroke laps therefore still get a positive elapsed when the next lap starts
    later; this matches native Garmin exports and avoids invalid / zero-duration laps in viewers.
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

    for j, lap_val in enumerate(interval_nrs):
        mask = (df[lap_col].values == lap_val)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        start_time_ms = int(unixtimes[indices[0]] * 1000)
        first_t = float(unixtimes[indices[0]])
        last_t = float(unixtimes[indices[-1]])
        if j + 1 < len(interval_nrs):
            next_lap_val = interval_nrs[j + 1]
            nmask = (df[lap_col].values == next_lap_val)
            nidx = np.where(nmask)[0]
            if len(nidx) > 0:
                total_elapsed_s = float(unixtimes[nidx[0]] - first_t)
            else:
                total_elapsed_s = float(last_t - first_t)
        else:
            total_elapsed_s = float(last_t - first_t)
        if total_elapsed_s < MIN_FIT_LAP_ELAPSED_S:
            total_elapsed_s = MIN_FIT_LAP_ELAPSED_S

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
        hr_seg = np.asarray(heart_rate[indices], dtype=float)
        hr_seg = hr_seg[np.isfinite(hr_seg)]
        max_hr = int(np.max(hr_seg)) if len(hr_seg) > 0 else 0

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


def _downsample_curve_series(series, n_points):
    """Downsample comma-separated curve to n_points. Returns list of floats per row."""
    curves = []
    for val in series:
        s = str(val).strip().strip('"')
        inner = s[1:-1] if len(s) > 2 else s
        parts = [float(x) for x in inner.split(',') if x.strip()]
        if len(parts) < 2:
            curves.append([0.0] * n_points)
            continue
        if len(parts) <= n_points:
            pad = [0.0] * (n_points - len(parts))
            curves.append(parts + pad)
        else:
            idx = np.linspace(0, len(parts) - 1, n_points, dtype=int)
            curves.append([float(parts[i]) for i in idx])
    return curves


def write_fit(file_name, df, row_date="2016-01-01", notes="Exported by Rowingdata",
              sport="rowing", use_developer_fields=True,
              instroke_export='off', instroke_columns=None, instroke_column_map=None,
              instroke_downsample_points=16, overwrite=True,
              instroke_abscissa_type=None, instroke_sample_interval_ms=None,
              garmin_parity_source_fit=None):
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
        For drive/peak force, **Newton** columns (`` AverageDriveForce (N)``,
        `` PeakDriveForce (N)``) are preferred; **lb** fields are still emitted when
        ``...(lbs)`` columns exist for backward compatibility (see ``fit_export_spec.json``).
    instroke_export : str
        'off' (default): no in-stroke curve export.
        'summary': export q1,q2,q3,q4,diff,maxpos,minpos per curve as developer fields.
        'downsampled': export fixed-length downsampled curve (SINT16 array) per stroke.
        'full': export full-resolution curve up to 127 points per stroke (FIT size limit).
        'companion': write curve data to .instroke.json sidecar file.
    instroke_columns : list, optional
        Curve columns to export. If None, auto-detect via _detect_instroke_columns.
    instroke_column_map : dict, optional
        Override mapping from df column name to canonical FIT curve type name.
        Default: curve_data->HandleForceCurve, boat accelerator curve->BoatAcceleratorCurve, etc.
    instroke_downsample_points : int
        For 'downsampled': number of points per stroke (default 16, range 2-127).
        Ignored for 'full' and other modes.
    instroke_abscissa_type : int or None
        X-axis semantics for in-stroke curves (developer fields 90-92). Use constants
        INSTROKE_ABSCISSA_* (0=unknown, 1=time uniform ms, ...). None = auto (time-based
        when `` DriveTime (ms)`` exists, else unknown).
    instroke_sample_interval_ms : float, array-like, or None
        Override per-stroke sample spacing for InstrokeSampleInterval (field 91); meaning
        depends on ``instroke_abscissa_type``. None = derive from drive time / point count.
    overwrite : bool
        If True (default), overwrite existing files. If False, raise FileExistsError
        when the target FIT file (or companion .instroke.json) already exists.
    garmin_parity_source_fit : str or None
        If set, path to a source FIT (e.g. Garmin / OpenRowingMonitor). After the Session
        message, native Workout, WorkoutStep, SplitSummary (mesg 313), and Split (mesg 312)
        data messages are re-emitted from that file via :mod:`rowingdata.fit_garmin_bridge`.
        Per-stroke data still comes from ``df`` and rowingdata developer field definitions.

    Returns
    -------
    dict or None
        None in the normal case. When notable conditions occur, returns a dict:
        - ``instroke_columns_available``: list of column names (when in-stroke data is
          detected but instroke_export='off')
        - ``suggestion``: hint to re-export with instroke_export enabled
        - ``companion_file``: path to .instroke.json (when instroke_export='companion')
    """
    if not FIT_TOOL_AVAILABLE:
        raise ImportError("fit-tool is required for FIT export. Install with: pip install fit-tool")

    # Detect in-stroke columns early (for return value and overwrite check)
    detected_instroke_cols = _detect_instroke_columns(df)
    curve_cols_for_export = (
        instroke_columns if instroke_columns is not None else detected_instroke_cols
    )
    instroke_would_write_companion = (
        instroke_export == 'companion' and
        [c for c in curve_cols_for_export if c in df.columns]
    )
    companion_path = None
    if instroke_would_write_companion:
        base, _ = os.path.splitext(file_name)
        companion_path = base + '.instroke.json'

    # Overwrite check: raise if overwrite=False and target(s) exist
    if not overwrite:
        if os.path.exists(file_name):
            raise FileExistsError(
                "FIT file already exists: %s. Set overwrite=True to replace." % file_name
            )
        if companion_path is not None and os.path.exists(companion_path):
            raise FileExistsError(
                "Companion file already exists: %s. Set overwrite=True to replace." % companion_path
            )

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

    has_gps = any(
        _valid_position(float(lat[i]), float(lon[i]))
        for i in range(nr_rows)
    ) if nr_rows > 0 else False

    # Stroke distance for native cycle_length16 field (UINT16, scale 100, max 655 m)
    stroke_distance = None
    if ' StrokeDistance (meters)' in df.columns:
        stroke_distance = np.nan_to_num(df[' StrokeDistance (meters)'].values, nan=0.0, posinf=0.0, neginf=0.0)
        stroke_distance = np.clip(stroke_distance, 0, 655.35)  # uint16 scale 100 max

    # Stroke number for native total_cycles field (data is per-stroke, one record per stroke)
    try:
        stroke_number = df[' Stroke Number'].values.astype(int)
    except KeyError:
        try:
            stroke_number = df['sessionStrokeIndex'].values.astype(int) + 1  # 1-based
        except KeyError:
            stroke_number = np.arange(1, nr_rows + 1, dtype=int)  # 1-based row index

    # Developer fields: which columns exist and their arrays (definitions from fit_export_spec.json)
    use_dev = use_developer_fields and FIT_TOOL_AVAILABLE
    dev_arrays = {}
    dev_specs = []
    DEV_DATA_IDX = 0
    if use_dev:
        _spec = fitwrite_spec.load_fit_spec()
        ROWING_DEV_FIELDS = _spec['ROWING_DEV_FIELDS']
        OARLOCK_DEV_FIELDS = _spec['OARLOCK_DEV_FIELDS']
        OARLOCK_DUAL_PAIRS = _spec['OARLOCK_DUAL_PAIRS']
        PEAK_POSITION_DEV_FIELDS = _spec['PEAK_POSITION_DEV_FIELDS']
        for fd in ROWING_DEV_FIELDS:
            field_id, possible_cols, name, base_type, size, scale, units = fd
            if not isinstance(possible_cols, (list, tuple)):
                possible_cols = [possible_cols]
            col = next((c for c in possible_cols if c in df.columns), None)
            if col is not None:
                arr = df[col].values
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                # Clip to avoid overflow: encoded = value * scale must fit in base_type
                if base_type == BaseType.UINT8:
                    arr = np.clip(arr, 0, 255)
                elif base_type == BaseType.UINT16:
                    max_display = 65535.0 / scale if scale else 65535.0
                    arr = np.clip(arr, 0, max_display)
                elif base_type == BaseType.SINT16:
                    max_display = 32767.0 / scale if scale else 32767.0
                    min_display = -32768.0 / scale if scale else -32768.0
                    arr = np.clip(arr, min_display, max_display)
                dev_arrays[field_id] = arr
                dev_specs.append((field_id, col, name, base_type, size, scale, units))
        for fd in OARLOCK_DEV_FIELDS:
            field_id, possible_cols, name, base_type, size, scale, units = fd
            col = next((c for c in possible_cols if c in df.columns), None)
            if col is not None:
                arr = df[col].values
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if base_type == BaseType.SINT16:
                    max_display = 32767.0 / scale if scale else 32767.0
                    min_display = -32768.0 / scale if scale else -32768.0
                    arr = np.clip(arr, min_display, max_display)
                elif base_type == BaseType.UINT16:
                    max_display = 65535.0 / scale if scale else 65535.0
                    arr = np.clip(arr, 0, max_display)
                dev_arrays[field_id] = arr
                dev_specs.append((field_id, col, name, base_type, size, scale, units))
        for _pair_key, port_fd, starboard_fd in OARLOCK_DUAL_PAIRS:
            cols_p = port_fd[1]
            cols_s = starboard_fd[1]
            col_p = next((c for c in cols_p if c in df.columns), None)
            col_s = next((c for c in cols_s if c in df.columns), None)
            if col_p is None or col_s is None:
                continue
            for fd, col in ((port_fd, col_p), (starboard_fd, col_s)):
                field_id, possible_cols, name, base_type, size, scale, units = fd
                arr = df[col].values
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                if base_type == BaseType.SINT16:
                    max_display = 32767.0 / scale if scale else 32767.0
                    min_display = -32768.0 / scale if scale else -32768.0
                    arr = np.clip(arr, min_display, max_display)
                elif base_type == BaseType.UINT16:
                    max_display = 65535.0 / scale if scale else 65535.0
                    arr = np.clip(arr, 0, max_display)
                dev_arrays[field_id] = arr
                dev_specs.append((field_id, col, name, base_type, size, scale, units))
        for fd in PEAK_POSITION_DEV_FIELDS:
            field_id, possible_cols, name, base_type, size, scale, units, transformer, clip_max = fd
            col = next((c for c in possible_cols if c in df.columns), None)
            if col is None:
                continue
            if transformer == 'peak_norm':
                arr = _series_to_peak_force_position_norm(df[col])
            elif transformer == 'peak_abs':
                arr = _series_to_peak_force_position_abs_m(df[col])
            else:
                continue
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if base_type == BaseType.UINT16:
                max_display = 65535.0 / scale if scale else 65535.0
                arr = np.clip(arr, 0, max_display)
                if clip_max is not None:
                    arr = np.clip(arr, 0, float(clip_max))
            dev_arrays[field_id] = arr
            dev_specs.append((field_id, col, name, base_type, size, scale, units))

    # In-stroke curve export (summary, downsampled, or companion)
    instroke_curve_cols = []
    instroke_summary_arrays = {}
    instroke_downsampled_arrays = {}
    col_map = instroke_column_map if instroke_column_map is not None else INSTROKE_COLUMN_MAP
    if instroke_export in ('summary', 'downsampled', 'full', 'companion'):
        curve_cols = instroke_columns if instroke_columns is not None else _detect_instroke_columns(df)
        instroke_curve_cols = [c for c in curve_cols if c in df.columns]
    if instroke_export == 'summary' and instroke_curve_cols and use_dev:
        base_id = _FIT_EXPORT_RAW['instroke_dynamic']['summary_start']
        for col in instroke_curve_cols:
            canonical = col_map.get(col, col)
            try:
                summ = _compute_instroke_summary(df, col)
            except (ValueError, KeyError, TypeError):
                continue
            for metric, arr in summ.items():
                field_id = base_id
                name = '%s_%s' % (canonical, metric)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                # UINT16: encoded = value * scale, must be <= 65535. Scale 1 => max display 65535
                scale = 1
                arr = np.clip(arr, 0, 65535.0 / max(scale, 1))
                dev_arrays[field_id] = arr
                dev_specs.append((field_id, col, name, BaseType.UINT16, 2, scale, ''))
                instroke_summary_arrays.setdefault(col, {})[metric] = field_id
                base_id += 1
            base_id = (base_id // 10 + 1) * 10
    elif instroke_export in ('downsampled', 'full') and instroke_curve_cols and use_dev:
        base_id = _FIT_EXPORT_RAW['instroke_dynamic']['curve_start']
        for col in instroke_curve_cols:
            canonical = col_map.get(col, col)
            try:
                curves_list, n_points = _get_instroke_curve_for_export(
                    df, col, instroke_export, instroke_downsample_points
                )
            except (ValueError, KeyError, TypeError):
                continue
            if n_points < 2:
                continue
            arr_2d = np.array(curves_list, dtype=np.float64)
            size = n_points * 2
            dev_arrays[base_id] = arr_2d
            dev_specs.append((base_id, col, canonical, BaseType.SINT16, size, 1, ''))
            base_id += 1

    if instroke_export in ('downsampled', 'full') and instroke_curve_cols and use_dev:
        nmax = 0
        for col in instroke_curve_cols:
            try:
                _, np_ = _get_instroke_curve_for_export(
                    df, col, instroke_export, instroke_downsample_points
                )
                nmax = max(nmax, np_)
            except (ValueError, KeyError, TypeError):
                pass
        if nmax >= 2:
            axis_tuples = _compute_instroke_axis_arrays(
                df, nmax, instroke_abscissa_type, instroke_sample_interval_ms)
            if axis_tuples is not None:
                ta, interval, pc = axis_tuples
                axis_arrays = [ta, interval, pc]
                for axis_row, arr in zip(
                        fitwrite_spec.load_fit_spec()['INSTROKE_AXIS_DEV_FIELDS'], axis_arrays):
                    fid, name, bt, size, scale, units = axis_row
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    if bt == BaseType.UINT8:
                        arr = np.clip(arr, 0, 255)
                    elif bt == BaseType.UINT16:
                        arr = np.clip(arr, 0, 65535)
                    dev_arrays[fid] = arr
                    dev_specs.append((fid, '_instroke_axis', name, bt, size, scale, units))

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
    session.sub_sport = _sub_sport_for_sport(sport, has_gps)
    if avg_hr > 0:
        session.avg_heart_rate = avg_hr
    if max_hr > 0:
        session.max_heart_rate = max_hr
    if avg_cadence > 0:
        session.avg_cadence = avg_cadence
    if avg_power > 0:
        session.avg_power = avg_power
    builder.add(session)

    if garmin_parity_source_fit:
        fit_garmin_bridge.add_preserved_messages_to_builder(builder, garmin_parity_source_fit)

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
                if field_id not in dev_arrays:
                    continue
                arr = dev_arrays[field_id]
                is_array_field = (base_type == BaseType.SINT16 and size > 2)
                if is_array_field and arr.ndim == 2:
                    row = arr[i]
                    vals = np.clip(row, -32768, 32767).astype(np.int32)
                    has_data = np.any(vals != 0)
                    if has_data:
                        dev = DeveloperField(
                            developer_data_index=DEV_DATA_IDX,
                            field_id=field_id,
                            size=size,
                            name=name,
                            base_type=base_type,
                            scale=1,
                            offset=0,
                            units=units
                        )
                        for j, v in enumerate(vals):
                            dev.set_value(j, int(v))
                        dev_fields.append(dev)
                else:
                    val = float(arr[i])
                    if base_type == BaseType.UINT8:
                        val = int(np.clip(round(val), 0, 255))
                    if val != 0 or field_id in ALWAYS_EMIT_DEV_FIELD_IDS:
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
        if stroke_distance is not None and hasattr(rec, 'cycle_length16'):
            rec.cycle_length16 = int(round(float(stroke_distance[i]) * 100))  # scale 100, cm precision
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
            lap.sub_sport = _sub_sport_for_sport(sport, has_gps)
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
        lap.sub_sport = _sub_sport_for_sport(sport, has_gps)
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

    # Companion export: write .instroke.json sidecar when instroke_export='companion'
    companion_written_path = None
    if instroke_export == 'companion' and instroke_curve_cols:
        col_map = instroke_column_map if instroke_column_map is not None else INSTROKE_COLUMN_MAP
        base, _ = os.path.splitext(file_name)
        companion_path = base + '.instroke.json'
        curves = {}
        for col in instroke_curve_cols:
            canonical = col_map.get(col, col)
            try:
                parsed = _parse_instroke_curve(df, col)
                # Per-stroke: list of curve values (as list for JSON)
                strokes = []
                for i in range(len(parsed)):
                    row = parsed.iloc[i].dropna().values
                    row = [float(x) for x in row if np.isfinite(x)]
                    strokes.append(row)
                curves[canonical] = strokes
            except (ValueError, KeyError, TypeError):
                pass
        if curves:
            max_pts = 0
            for strokes in curves.values():
                for row in strokes:
                    max_pts = max(max_pts, len(row))
            max_pts = max(max_pts, 2)
            axis_tuples = _compute_instroke_axis_arrays(
                df, max_pts, instroke_abscissa_type, instroke_sample_interval_ms)
            meta = {
                'version': 1,
                'instroke_abscissa_type': (
                    int(axis_tuples[0][0]) if axis_tuples is not None
                    else INSTROKE_ABSCISSA_UNKNOWN
                ),
                'instroke_point_count': int(max_pts),
            }
            if axis_tuples is not None:
                _, interval, _ = axis_tuples
                meta['instroke_sample_interval_ms'] = [float(x) for x in interval]
            out = dict(curves)
            out['_rowingdata_instroke'] = meta
            with open(companion_path, 'w') as f:
                json.dump(out, f)
            companion_written_path = companion_path

    # Build return value for notable conditions
    result = None
    if instroke_export == 'off' and detected_instroke_cols:
        result = {
            'instroke_columns_available': detected_instroke_cols,
            'suggestion': 'Re-export with instroke_export="summary", "downsampled", "full", or "companion" to include curve data.'
        }
    if companion_written_path is not None:
        if result is None:
            result = {}
        result['companion_file'] = companion_written_path

    return result
