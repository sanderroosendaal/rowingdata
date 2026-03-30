# -*- coding: utf-8 -*-
"""
Transcode Garmin / OpenRowingMonitor-style FIT files into rowingdata FIT exports.

Reads per-stroke records with ORM-style developer field names and maps them to
rowingdata DataFrame columns for :func:`rowingdata.fitwrite.write_fit`.
Optional ``garmin_parity_source_fit`` preserves native Workout, WorkoutStep,
Split (mesg 312), and SplitSummary (mesg 313) messages from the source file.
"""
from __future__ import absolute_import

import numpy as np
import pandas as pd

# Native + ORM developer names from Garmin indoor rowing (see field_description CSV)
_GARMIN_DEV_TO_ROWINGDATA = {
    'DragFactor': ' DragFactor',
    'StrokeDriveTime': ' DriveTime (ms)',
    'StrokeRecoveryTime': ' StrokeRecoveryTime (ms)',
    'DriveLength': ' DriveLength (meters)',
    'AverageDriveForce': ' AverageDriveForce (N)',
    'PeakDriveForce': ' PeakDriveForce (N)',
    'PeakForceNormPosition': 'PeakForcePositionNorm',  # rowingdata peak norm column
    'HandleForceCurveXAxis': '_orm_HandleForceCurveXAxis',
    'HandleForceCurveLength': '_orm_HandleForceCurveLength',
    'HandleForceCurve': 'curve_data',
}


def data_frame_from_garmin_fit(fit_path):
    """
    Build a rowingdata-compatible DataFrame from a Garmin/ORM FIT with per-stroke records.

    Expects ``record`` messages with power, cadence, distance, and optional developer
    fields matching the ORM naming convention. In-stroke curve is stored as
    ``curve_data`` in comma-separated list form (same as RP3).
    """
    from fitparse import FitFile

    f = FitFile(fit_path)
    rows = []
    for m in f.get_messages('record'):
        d = {}
        for field in m.fields:
            name = field.name
            val = field.value
            if name == 'timestamp' and val is not None:
                # fitparse datetime -> seconds since epoch
                d['_ts'] = val
                continue
            if name in ('distance', 'heart_rate', 'cadence', 'power', 'enhanced_speed',
                         'speed', 'total_cycles', 'accumulated_power', 'resistance'):
                d[name] = val
                if name == 'total_cycles' and val is not None:
                    d[' Stroke Number'] = int(val) + 1  # FIT total_cycles 0-based -> 1-based
                continue
            if name in _GARMIN_DEV_TO_ROWINGDATA:
                rd = _GARMIN_DEV_TO_ROWINGDATA[name]
                if name == 'HandleForceCurve':
                    # tuple of ints -> comma-separated string for curve_data
                    if isinstance(val, (list, tuple)):
                        d[rd] = '(' + ','.join(str(int(x)) for x in val) + ')'
                    else:
                        d[rd] = val
                elif name == 'PeakForceNormPosition':
                    # ORM uint8 percent 0–100 -> PeakForcePositionNorm ten-thousandths 0–10000
                    v = float(val) if val is not None else 0.0
                    d[rd] = v * 100.0 if v <= 100.0 else v
                else:
                    d[rd] = val
        rows.append(d)

    if not rows:
        raise ValueError('No record messages in FIT: %s' % fit_path)

    df = pd.DataFrame(rows)
    # timestamps
    if '_ts' in df.columns:
        base = df['_ts'].iloc[0]
        if hasattr(base, 'timestamp'):
            df['TimeStamp (sec)'] = df['_ts'].apply(lambda t: t.timestamp() if hasattr(t, 'timestamp') else float(t))
        else:
            df['TimeStamp (sec)'] = df['_ts'].astype(float)
        df = df.drop(columns=['_ts'], errors='ignore')

    # cum_dist / horizontal
    if 'distance' in df.columns:
        df['cum_dist'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)
        df[' Horizontal (meters)'] = df['cum_dist']

    if 'cadence' in df.columns:
        df[' Cadence (stokes/min)'] = pd.to_numeric(df['cadence'], errors='coerce').fillna(0)
    if 'heart_rate' in df.columns:
        df[' HRCur (bpm)'] = pd.to_numeric(df['heart_rate'], errors='coerce').fillna(0)
    if 'power' in df.columns:
        df[' Power (watts)'] = pd.to_numeric(df['power'], errors='coerce').fillna(0)
    if 'enhanced_speed' in df.columns:
        df['Stroke500mPace (sec/500m)'] = np.where(
            df['enhanced_speed'].astype(float) > 0,
            500.0 / df['enhanced_speed'].astype(float),
            0.0,
        )

    # Drop native columns we mapped
    for c in ('distance', 'cadence', 'heart_rate', 'power', 'enhanced_speed', 'speed',
              'total_cycles', 'accumulated_power', 'resistance', 'activity_type'):
        if c in df.columns:
            df = df.drop(columns=[c])

    # Drop ORM-only helper columns not used by rowingdata FIT spec
    df = df.drop(columns=[c for c in df.columns if c.startswith('_orm_')], errors='ignore')
    return df


def transcode_garmin_fit_to_rowingdata(
    source_fit_path,
    dest_fit_path,
    row_date=None,
    sport='indoor_rowing',
    use_developer_fields=True,
    instroke_export='full',
    **kwargs
):
    """
    Read a Garmin/ORM FIT, map to rowingdata columns, write FIT with
    rowingdata developer spec + preserved Split/Workout messages from source.

    Parameters
    ----------
    source_fit_path : str
        Input FIT (e.g. OpenRowingMonitor export).
    dest_fit_path : str
        Output path.
    row_date : str or None
        If None, inferred from first record timestamp.
    **kwargs
        Passed to :func:`rowingdata.fitwrite.write_fit` (e.g. `instroke_downsample_points`).
    """
    from .fitwrite import write_fit

    df = data_frame_from_garmin_fit(source_fit_path)
    if row_date is None and 'TimeStamp (sec)' in df.columns:
        t0 = df['TimeStamp (sec)'].iloc[0]
        import datetime
        row_date = datetime.datetime.utcfromtimestamp(float(t0)).strftime('%Y-%m-%d')

    return write_fit(
        dest_fit_path,
        df,
        row_date=row_date or '2016-01-01',
        sport=sport,
        use_developer_fields=use_developer_fields,
        instroke_export=instroke_export,
        garmin_parity_source_fit=source_fit_path,
        **kwargs
    )
