# -*- coding: utf-8 -*-
"""
Transcode Garmin / OpenRowingMonitor-style FIT files into rowingdata FIT exports.

Uses :class:`rowingdata.otherparsers.FITParser` for the base DataFrame (including
`` lapIdx`` from lap message timestamps) and maps ORM-style developer field names
to rowingdata columns for :func:`rowingdata.fitwrite.write_fit`.

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


def _norm_col_key(name):
    return ''.join(str(name).strip().lower().split())


def data_frame_from_garmin_fit(fit_path):
    """
    Build a rowingdata-compatible DataFrame from a Garmin/ORM FIT with per-stroke records.

    Uses :class:`rowingdata.otherparsers.FITParser` so `` lapIdx`` is derived from
    Lap ``start_time`` / ``timestamp`` (not message order). Maps ORM developer field
    names to rowingdata columns. In-stroke curve is stored as ``curve_data`` in
    comma-separated list form (same as RP3).
    """
    from .otherparsers import FITParser

    p = FITParser(fit_path)
    df = p.df.copy()

    # Rename ORM / Garmin developer columns to rowingdata names (case-insensitive)
    rename = {}
    for c in list(df.columns):
        ck = _norm_col_key(c)
        for garmin_key, rd_col in _GARMIN_DEV_TO_ROWINGDATA.items():
            if ck == _norm_col_key(garmin_key):
                if c != rd_col:
                    rename[c] = rd_col
                break
    if rename:
        df = df.rename(columns=rename)

    if 'PeakForcePositionNorm' in df.columns:
        def _peak_norm(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return v
            try:
                x = float(v)
            except (TypeError, ValueError):
                return v
            return x * 100.0 if x <= 100.0 else x

        df['PeakForcePositionNorm'] = df['PeakForcePositionNorm'].apply(_peak_norm)

    if 'curve_data' in df.columns:
        def _fmt_curve(v):
            if isinstance(v, (list, tuple)):
                return '(' + ','.join(str(int(x)) for x in v) + ')'
            return v

        df['curve_data'] = df['curve_data'].apply(_fmt_curve)

    df = df.drop(columns=[c for c in df.columns if str(c).startswith('_orm_')], errors='ignore')
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
