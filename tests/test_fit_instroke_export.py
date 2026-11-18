# -*- coding: utf-8 -*-
"""Thorough tests for FIT in-stroke curve export.

Only HandleForceCurve (60) and axis fields 90-92 go in the FIT file, per the Rowing Data Standard
(https://github.com/Rowing-Industry-Trade-Association/rowing-data-standard); other curves go to the
companion JSON file.
"""
from __future__ import absolute_import

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from fitparse import FitFile

from rowingdata import fitwrite
from rowingdata import fitwrite_spec
from rowingdata.fitwrite import (
    INSTROKE_ABSCISSA_HANDLE_DISTANCE_UNIFORM_M,
    INSTROKE_ABSCISSA_TIME_UNIFORM_MS,
    RECORDING_STRATEGY_GPS_UPDATE,
)


def _field_description_map(fit_path):
    """Map field_definition_number -> field_description message fields."""
    by_id = {}
    for msg in FitFile(fit_path).get_messages('field_description'):
        row = {f.name: f.value for f in msg}
        by_id[row['field_definition_number']] = row
    return by_id


def _record_field_maps(fit_path):
    """List of dicts: field name -> fitparse FieldData for each record message."""
    out = []
    for msg in FitFile(fit_path).get_messages('record'):
        out.append({f.name: f for f in msg})
    return out


def _synthetic_stroke_df(
    curve_values=(100, 200, 400, 300, 50),
    drive_length_m=1.0,
    drive_time_ms=500,
    n_strokes=1,
):
    curve_str = '(' + ','.join(str(int(v)) for v in curve_values) + ')'
    n = max(n_strokes, 1)
    return pd.DataFrame({
        'TimeStamp (sec)': np.arange(1.0, n + 1.0, 1.0),
        ' Horizontal (meters)': np.arange(0.0, n * 10.0, 10.0),
        ' Cadence (stokes/min)': np.full(n, 20.0),
        ' DriveLength (meters)': np.full(n, drive_length_m),
        ' DriveTime (ms)': np.full(n, drive_time_ms),
        'curve_data': [curve_str] * n,
    })


def _export_synthetic(tmp_path, instroke_export, **kwargs):
    df = kwargs.pop('df', None)
    if df is None:
        df = _synthetic_stroke_df()
    fit_path = os.path.join(tmp_path, 'instroke_test.fit')
    fitwrite.write_fit(
        fit_path,
        df,
        row_date='2026-08-03',
        sport='rowing',
        use_developer_fields=True,
        instroke_export=instroke_export,
        overwrite=True,
        **kwargs
    )
    return fit_path, df


class TestFitInstrokeHelpers(unittest.TestCase):
    def test_field_description_map_collects_developer_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(tmp, 'downsampled', instroke_downsample_points=5)
            fmap = _field_description_map(fit_path)
            self.assertIn(60, fmap)
            self.assertEqual(fmap[60]['field_name'], 'HandleForceCurve')


class TestFitInstrokeExportPhase1(unittest.TestCase):
    """Synthetic fixture + fitparse: axis 90–92, HandleForceCurve ID 60, Y scale."""

    def test_handle_force_curve_field_description_id_and_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=5,
            )
            fmap = _field_description_map(fit_path)
            self.assertEqual(fmap[60]['field_name'], 'HandleForceCurve')
            self.assertEqual(fmap[60]['scale'], 10)
            self.assertEqual(fmap[60]['units'], 'N')
            self.assertEqual(fmap[60]['fit_base_type_id'], 'uint16')

    def test_axis_field_descriptions_ids_90_to_92(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=5,
            )
            fmap = _field_description_map(fit_path)
            self.assertEqual(fmap[90]['field_name'], 'InstrokeAbscissaType')
            self.assertEqual(fmap[91]['field_name'], 'InstrokeSampleInterval')
            self.assertEqual(fmap[92]['field_name'], 'InstrokePointCount')

    def test_default_abscissa_handle_distance_uniform_mm_interval(self):
        """DriveLength present → type 2, interval in mm = drive_mm / (n-1)."""
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=5,
            )
            rec = _record_field_maps(fit_path)[0]
            self.assertEqual(
                rec['InstrokeAbscissaType'].raw_value,
                INSTROKE_ABSCISSA_HANDLE_DISTANCE_UNIFORM_M,
            )
            self.assertEqual(rec['InstrokePointCount'].raw_value, 5)
            # 1.0 m drive, 5 points → 1000 mm / 4 = 250 mm per step
            self.assertEqual(rec['InstrokeSampleInterval'].raw_value, 250)

    def test_explicit_time_abscissa_interval_ms(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(
                tmp,
                'downsampled',
                instroke_downsample_points=5,
                instroke_abscissa_type=INSTROKE_ABSCISSA_TIME_UNIFORM_MS,
            )
            rec = _record_field_maps(fit_path)[0]
            self.assertEqual(
                rec['InstrokeAbscissaType'].raw_value,
                INSTROKE_ABSCISSA_TIME_UNIFORM_MS,
            )
            # 500 ms drive, 5 points → 500/4 = 125 ms
            self.assertEqual(rec['InstrokeSampleInterval'].raw_value, 125)

    def test_handle_force_curve_raw_values_match_source_newtons(self):
        """Encoded curve = force (N) × Y scale (10)."""
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=5,
            )
            rec = _record_field_maps(fit_path)[0]
            raw = rec['HandleForceCurve'].raw_value
            self.assertEqual(raw, (1000, 2000, 4000, 3000, 500))

    def test_instroke_fields_absent_when_export_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(tmp, 'off')
            fmap = _field_description_map(fit_path)
            self.assertNotIn(60, fmap)
            self.assertNotIn(90, fmap)
            recs = _record_field_maps(fit_path)
            self.assertNotIn('HandleForceCurve', recs[0])

    def test_summary_mode_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _export_synthetic(tmp, 'summary')

    def test_spec_curve_start_and_axis_ids(self):
        raw = fitwrite_spec.load_fit_spec_raw()
        self.assertEqual(raw['instroke_dynamic']['curve_start'], 60)
        self.assertEqual(list(raw['instroke_axis_field_ids']), [90, 91, 92])
        handle = raw['instroke_curve_types']['HandleForceCurve']
        self.assertEqual(handle['y_scale'], 10)
        self.assertEqual(handle['default_abscissa'], 'HANDLE_DISTANCE_UNIFORM_M')
        # Only HandleForceCurve is a FIT curve; others live in the companion JSON.
        self.assertEqual(list(raw['instroke_curve_types']), ['HandleForceCurve'])
        self.assertNotIn('summary_start', raw['instroke_dynamic'])


class TestFitInstrokeExportPhase2(unittest.TestCase):
    """Companion fidelity, downsampled point counts."""

    def test_downsampled_short_curve_is_not_padded(self):
        """A curve shorter than the requested count is written at its own length."""
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=16,
            )
            rec = _record_field_maps(fit_path)[0]
            self.assertEqual(rec['InstrokePointCount'].raw_value, 5)
            self.assertEqual(rec['HandleForceCurve'].raw_value, (1000, 2000, 4000, 3000, 500))

    def test_downsampled_long_curve_is_reduced_to_requested_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = _synthetic_stroke_df(curve_values=tuple(range(10, 410, 10)))  # 40 values
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', df=df, instroke_downsample_points=16,
            )
            rec = _record_field_maps(fit_path)[0]
            self.assertEqual(rec['InstrokePointCount'].raw_value, 16)
            self.assertEqual(len(rec['HandleForceCurve'].raw_value), 16)

    def test_interval_follows_each_records_own_point_count(self):
        """Interval mm = drive_mm / (point_count - 1) of that record's curve."""
        with tempfile.TemporaryDirectory() as tmp:
            # 5 real samples over 1000 mm -> 250 mm
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=16,
            )
            rec = _record_field_maps(fit_path)[0]
            self.assertEqual(rec['InstrokeSampleInterval'].raw_value, 250)
            # 40 samples reduced to 16 points over 1000 mm -> 1000 / 15 = 67 mm
            df = _synthetic_stroke_df(curve_values=tuple(range(10, 410, 10)))
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', df=df, instroke_downsample_points=16,
            )
            rec = _record_field_maps(fit_path)[0]
            self.assertEqual(rec['InstrokeSampleInterval'].raw_value, 67)

    def test_companion_json_preserves_full_source_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = _synthetic_stroke_df(
                curve_values=(100, 200, 400, 300, 50),
                n_strokes=2,
            )
            df.loc[1, 'curve_data'] = '(10,20,30,40,50)'
            fit_path = os.path.join(tmp, 'companion.fit')
            fitwrite.write_fit(
                fit_path,
                df,
                row_date='2026-08-03',
                sport='rowing',
                instroke_export='companion',
                overwrite=True,
            )
            companion = os.path.splitext(fit_path)[0] + '.instroke.json'
            self.assertTrue(os.path.isfile(companion))
            with open(companion) as f:
                data = json.load(f)
            strokes = data['HandleForceCurve']
            self.assertEqual(len(strokes), 2)
            self.assertEqual(strokes[0], [100.0, 200.0, 400.0, 300.0, 50.0])
            self.assertEqual(strokes[1], [10.0, 20.0, 30.0, 40.0, 50.0])

    def test_companion_metadata_abscissa_and_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, df = _export_synthetic(tmp, 'companion')
            companion = os.path.splitext(fit_path)[0] + '.instroke.json'
            with open(companion) as f:
                data = json.load(f)
            meta = data['_rowingdata_instroke']
            self.assertEqual(meta['version'], 1)
            self.assertEqual(
                meta['instroke_abscissa_type'],
                INSTROKE_ABSCISSA_HANDLE_DISTANCE_UNIFORM_M,
            )
            self.assertEqual(meta['instroke_point_count'], 5)
            self.assertEqual(len(meta['instroke_sample_interval_ms']), len(df))
            self.assertEqual(meta['instroke_sample_interval_ms'][0], 250.0)

    def test_companion_no_fit_curve_arrays(self):
        """Companion mode writes JSON only; FIT should not contain HandleForceCurve arrays."""
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(tmp, 'companion')
            fmap = _field_description_map(fit_path)
            self.assertNotIn(60, fmap)
            rec = _record_field_maps(fit_path)[0]
            self.assertNotIn('HandleForceCurve', rec)


def _parse_curve_parenthesized(value):
    """Parse '(a,b,c)' curve string or return list from numeric sequence."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text.startswith('('):
        return [float(text)]
    inner = text[1:-1]
    if not inner:
        return []
    return [float(x.strip()) for x in inner.split(',') if x.strip()]


class TestFitInstrokeExportPhase3(unittest.TestCase):
    """GPS_UPDATE guard, Quiske multi-curve, golden example, FITParser round-trip."""

    def test_gps_update_rejects_instroke_curve_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = _synthetic_stroke_df()
            fit_path = os.path.join(tmp, 'gps_update.fit')
            with self.assertRaises(ValueError) as ctx:
                fitwrite.write_fit(
                    fit_path,
                    df,
                    row_date='2026-08-03',
                    sport='rowing',
                    instroke_export='full',
                    recording_strategy=RECORDING_STRATEGY_GPS_UPDATE,
                    overwrite=True,
                )
            self.assertIn('GPS_UPDATE', str(ctx.exception))

    def test_quiske_non_force_curves_go_to_companion_file(self):
        """Standard section 6.1: only HandleForceCurve is written to the FIT file."""
        import json
        import rowingdata
        csvfile = 'testdata/quiske_per_stroke_left.csv'
        with tempfile.TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, 'quiske_instroke.fit')
            r = rowingdata.QuiskeParser(csvfile)
            row = rowingdata.rowingdata(df=r.df, absolutetimestamps=False)
            row.exporttofit(
                outfile,
                sport='rowing',
                instroke_export='downsampled',
                instroke_downsample_points=16,
            )
            fmap = _field_description_map(outfile)
            for fid in (60, 61):
                self.assertNotIn(fid, fmap)
            names = [f['field_name'] for f in fmap.values()]
            for name in ('BoatAcceleratorCurve', 'OarAngleVelocityCurve'):
                self.assertNotIn(name, names)
            rec = _record_field_maps(outfile)[0]
            self.assertNotIn('BoatAcceleratorCurve', rec)
            self.assertNotIn('InstrokePointCount', rec)
            companion = os.path.join(tmp, 'quiske_instroke.instroke.json')
            self.assertTrue(os.path.isfile(companion))
            with open(companion) as f:
                data = json.load(f)
            self.assertIn('BoatAcceleratorCurve', data)
            self.assertIn('OarAngleVelocityCurve', data)

    def test_golden_standard_example_instroke_metadata_and_samples(self):
        path = 'testdata/rowingdata_standard_example.fit'
        fmap = _field_description_map(path)
        self.assertEqual(fmap[60]['field_name'], 'HandleForceCurve')
        self.assertEqual(fmap[60]['scale'], 10)
        self.assertEqual(fmap[60]['units'], 'N')
        for rec in _record_field_maps(path):
            if 'HandleForceCurve' not in rec:
                continue
            self.assertEqual(
                rec['InstrokeAbscissaType'].raw_value,
                INSTROKE_ABSCISSA_TIME_UNIFORM_MS,
            )
            self.assertEqual(rec['InstrokePointCount'].raw_value, 16)
            self.assertEqual(len(rec['HandleForceCurve'].raw_value), 16)
            self.assertEqual(
                rec['HandleForceCurve'].raw_value[:5],
                (30, 2880, 4510, 4840, 5090),
            )
            break
        else:
            self.fail('golden example FIT has no HandleForceCurve records')

    def test_fitparser_handle_force_curve_roundtrip(self):
        import rowingdata
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, df = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=5,
            )
            parsed = rowingdata.FITParser(fit_path)
            self.assertIn('curve_data', parsed.df.columns)
            values = _parse_curve_parenthesized(parsed.df['curve_data'].iloc[0])
            self.assertEqual(values, [100.0, 200.0, 400.0, 300.0, 50.0])
            source = _parse_curve_parenthesized(df['curve_data'].iloc[0])
            self.assertEqual(values, source)


def _curve_values(n_points, peak):
    """Smooth force-curve-like profile (half sine, in N) with ``n_points`` samples."""
    x = np.linspace(0.0, np.pi, n_points)
    return [round(float(v), 1) for v in peak * np.sin(x) + 5.0]


class TestFitInstrokeOrmHandleForce(unittest.TestCase):
    """OpenRowingMonitor names its in-stroke force column ``Handle_Force_(N)``."""

    ORM_COLUMN = 'Handle_Force_(N)'

    def _orm_df(self):
        df = _synthetic_stroke_df(curve_values=(100, 200, 400, 300, 50), n_strokes=3)
        df = df.rename(columns={'curve_data': self.ORM_COLUMN})
        # ORM writes the curve without brackets
        df[self.ORM_COLUMN] = ['100,200,400,300,50'] * len(df)
        df['Handle_Velocity_(m/s)'] = ['0.5,1.0,1.5,1.0,0.5'] * len(df)
        df['Handle_Power_(W)'] = ['50,200,600,300,25'] * len(df)
        return df

    def test_column_map_names_orm_column(self):
        self.assertEqual(fitwrite.INSTROKE_COLUMN_MAP[self.ORM_COLUMN], 'HandleForceCurve')

    def test_orm_handle_force_goes_to_fit_field_60(self):
        for mode in ('full', 'downsampled'):
            with tempfile.TemporaryDirectory() as tmp:
                fit_path, _ = _export_synthetic(tmp, mode, df=self._orm_df())
                fmap = _field_description_map(fit_path)
                self.assertEqual(fmap[60]['field_name'], 'HandleForceCurve')
                recs = [r for r in _record_field_maps(fit_path) if 'HandleForceCurve' in r]
                self.assertEqual(len(recs), 3)
                for rec in recs:
                    self.assertEqual(list(rec['HandleForceCurve'].raw_value),
                                     [1000, 2000, 4000, 3000, 500])
                    self.assertEqual(rec['InstrokePointCount'].raw_value, 5)

    def test_orm_handle_force_not_in_companion_when_in_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(tmp, 'full', df=self._orm_df())
            with open(os.path.splitext(fit_path)[0] + '.instroke.json') as f:
                companion = json.load(f)
            self.assertNotIn(self.ORM_COLUMN, companion)
            self.assertNotIn('HandleForceCurve', companion)

    def test_orm_velocity_and_power_stay_out_of_fit(self):
        """The standard (§6.1) allows only the handle force curve in the FIT file."""
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(tmp, 'full', df=self._orm_df())
            names = {d['field_name'] for d in _field_description_map(fit_path).values()}
            self.assertNotIn('Handle_Velocity_(m/s)', names)
            self.assertNotIn('Handle_Power_(W)', names)
            with open(os.path.splitext(fit_path)[0] + '.instroke.json') as f:
                companion = json.load(f)
            self.assertIn('Handle_Velocity_(m/s)', companion)
            self.assertIn('Handle_Power_(W)', companion)


class TestFitInstrokeNativeCsvFullResolution(unittest.TestCase):
    """Native rowingdata CSV with 50-100 point force curves -> FIT at original resolution."""

    POINTS_PER_STROKE = [50, 64, 75, 99, 100]
    STROKE_DISTANCE_M = [9.87, 10.05, 10.31, 9.5, 10.0]

    def _write_native_csv(self, path):
        n = len(self.POINTS_PER_STROKE)
        curves = [_curve_values(p, 300.0 + 20 * i) for i, p in enumerate(self.POINTS_PER_STROKE)]
        df = pd.DataFrame({
            'TimeStamp (sec)': 1500000000.0 + np.arange(1.0, n + 1.0) * 3.0,
            ' lapIdx': np.zeros(n, dtype=int),
            ' Horizontal (meters)': np.arange(1, n + 1) * 10.0,
            ' Cadence (stokes/min)': np.full(n, 24.0),
            ' Stroke500mPace (sec/500m)': np.full(n, 120.0),
            ' Power (watts)': np.full(n, 200.0),
            ' HRCur (bpm)': np.full(n, 140),
            ' DriveLength (meters)': np.full(n, 1.2),
            ' DriveTime (ms)': np.full(n, 900),
            ' StrokeDistance (meters)': self.STROKE_DISTANCE_M,
            'curve_data': ['(' + ','.join('%g' % v for v in c) + ')' for c in curves],
        })
        df.to_csv(path, index=False)
        return curves

    def test_native_csv_curves_exported_at_original_resolution(self):
        import rowingdata
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'native_curves.csv')
            fit_path = os.path.join(tmp, 'native_curves.fit')
            source = self._write_native_csv(csv_path)

            row = rowingdata.rowingdata(csvfile=csv_path)
            self.assertIn('curve_data', row.df.columns)
            row.exporttofit(fit_path, sport='rowing', instroke_export='full')

            # Field description: HandleForceCurve, UINT16 array, scale 10, N
            fmap = _field_description_map(fit_path)
            self.assertEqual(fmap[60]['field_name'], 'HandleForceCurve')
            self.assertEqual(fmap[60]['scale'], 10)
            self.assertEqual(fmap[60]['units'], 'N')

            recs = [r for r in _record_field_maps(fit_path) if 'HandleForceCurve' in r]
            self.assertEqual(len(recs), len(source))

            for rec, curve, n_points in zip(recs, source, self.POINTS_PER_STROKE):
                raw = rec['HandleForceCurve'].raw_value
                # Original resolution: exactly the source samples, no padding, no resampling
                self.assertEqual(rec['InstrokePointCount'].raw_value, n_points)
                self.assertEqual(list(raw), [int(round(v * 10)) for v in curve])
                self.assertLessEqual(len(raw), fitwrite.INSTROKE_MAX_POINTS)
                # Standard §6.3: axis metadata describes this record's curve
                self.assertEqual(
                    rec['InstrokeAbscissaType'].raw_value,
                    INSTROKE_ABSCISSA_HANDLE_DISTANCE_UNIFORM_M)
                # drive length 1.2 m = 1200 mm spread over n_points - 1 intervals
                self.assertEqual(
                    rec['InstrokeSampleInterval'].raw_value,
                    int(round(1200.0 / (n_points - 1))))

    def test_native_csv_cycle_length16_written_per_record(self):
        """Record field 87 (cycle_length16, UINT16, scale 100 m) = distance per stroke."""
        import rowingdata
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'native_curves.csv')
            fit_path = os.path.join(tmp, 'native_curves.fit')
            self._write_native_csv(csv_path)
            row = rowingdata.rowingdata(csvfile=csv_path)
            row.exporttofit(fit_path, sport='rowing', instroke_export='full')

            raw = [
                {f.name: f.raw_value for f in m}.get('unknown_87')  # fitparse lacks 87
                for m in FitFile(fit_path).get_messages('record')
            ]
            self.assertEqual(raw, [int(round(d * 100)) for d in self.STROKE_DISTANCE_M])

            parsed = rowingdata.FITParser(fit_path)
            np.testing.assert_allclose(
                parsed.df[' StrokeDistance (meters)'].values, self.STROKE_DISTANCE_M)

    def test_native_csv_curves_survive_fit_roundtrip(self):
        import rowingdata
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'native_curves.csv')
            fit_path = os.path.join(tmp, 'native_curves.fit')
            source = self._write_native_csv(csv_path)
            row = rowingdata.rowingdata(csvfile=csv_path)
            row.exporttofit(fit_path, sport='rowing', instroke_export='full')

            parsed = rowingdata.FITParser(fit_path)
            self.assertIn('curve_data', parsed.df.columns)
            curves = [
                _parse_curve_parenthesized(v)
                for v in parsed.df['curve_data'] if isinstance(v, str)
            ]
            self.assertEqual(len(curves), len(source))
            for got, want in zip(curves, source):
                self.assertEqual(got, want)
