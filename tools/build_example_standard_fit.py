#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenerate testdata/rowingdata_standard_example.fit — a reference FIT aligned with
rowingdata/data/fit_export_spec.json (developer fields, multi-lap, downsampled in-stroke).

Requires: fit-tool (same as rowingdata FIT export).

Usage (from repo root):
    python tools/build_example_standard_fit.py
"""
from __future__ import absolute_import

import datetime
import os
import sys

import pandas as pd

# Repo root on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from rowingdata import RowPerfectParser, rowingdata
    from rowingdata.fitwrite import INSTROKE_ABSCISSA_TIME_UNIFORM_MS

    csv_path = os.path.join(_ROOT, 'testdata', 'rp3intervals2.csv')
    out_path = os.path.join(_ROOT, 'testdata', 'rowingdata_standard_example.fit')

    # Fixed timestamps for reproducible exports (matches other tests using RP3 sample).
    row_date = datetime.datetime(2016, 7, 28, 9, 35, 29)

    r = RowPerfectParser(csv_path, row_date=row_date)
    raw = pd.read_csv(csv_path).sort_values(['workout_interval_id', 'stroke_number'])
    lap_idx, _ = pd.factorize(raw['workout_interval_id'].values)
    if len(lap_idx) != len(r.df):
        raise SystemExit('Row count mismatch between CSV and RowPerfectParser')
    r.df[' lapIdx'] = lap_idx

    row = rowingdata(df=r.df, absolutetimestamps=False)
    row.exporttofit(
        out_path,
        sport='indoor_rowing',
        notes='rowingdata FIT export spec reference example (testdata/rowingdata_standard_example.fit)',
        instroke_export='downsampled',
        instroke_downsample_points=16,
        instroke_abscissa_type=INSTROKE_ABSCISSA_TIME_UNIFORM_MS,
        overwrite=True,
    )
    print('Wrote', out_path, '(%d bytes)' % os.path.getsize(out_path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
