# -*- coding: utf-8 -*-
"""Thorough tests for FIT in-stroke curve export (FIT_STANDARD §6)."""
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

    def test_spec_curve_start_and_axis_ids(self):
        raw = fitwrite_spec.load_fit_spec_raw()
        self.assertEqual(raw['instroke_dynamic']['curve_start'], 60)
        self.assertEqual(list(raw['instroke_axis_field_ids']), [90, 91, 92])
        handle = raw['instroke_curve_types']['HandleForceCurve']
        self.assertEqual(handle['y_scale'], 10)
        self.assertEqual(handle['default_abscissa'], 'HANDLE_DISTANCE_UNIFORM_M')


class TestFitInstrokeExportPhase2(unittest.TestCase):
    """Summary IDs/names, companion fidelity, downsampled point counts."""

    def test_summary_field_ids_20_through_26(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(tmp, 'summary')
            fmap = _field_description_map(fit_path)
            expected = {
                20: 'HandleForceCurve_q1',
                21: 'HandleForceCurve_q2',
                22: 'HandleForceCurve_q3',
                23: 'HandleForceCurve_q4',
                24: 'HandleForceCurve_diff',
                25: 'HandleForceCurve_maxpos',
                26: 'HandleForceCurve_minpos',
            }
            for fid, name in expected.items():
                self.assertIn(fid, fmap)
                self.assertEqual(fmap[fid]['field_name'], name)

    def test_summary_mode_no_axis_field_descriptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(tmp, 'summary')
            fmap = _field_description_map(fit_path)
            self.assertNotIn(90, fmap)
            self.assertNotIn(91, fmap)
            self.assertNotIn(92, fmap)
            rec = _record_field_maps(fit_path)[0]
            self.assertNotIn('InstrokeAbscissaType', rec)

    def test_summary_metrics_follow_computed_zero_omission_rule(self):
        """Records include summary dev fields only when computed metric != 0."""
        with tempfile.TemporaryDirectory() as tmp:
            df = _synthetic_stroke_df()
            summ = fitwrite._compute_instroke_summary(df, 'curve_data')
            fit_path, _ = _export_synthetic(tmp, 'summary', df=df)
            rec = _record_field_maps(fit_path)[0]
            for metric, arr in summ.items():
                key = 'HandleForceCurve_%s' % metric
                if arr[0] == 0:
                    self.assertNotIn(key, rec)
                else:
                    self.assertIn(key, rec)

    def test_downsampled_point_count_16_pads_short_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=16,
            )
            rec = _record_field_maps(fit_path)[0]
            self.assertEqual(rec['InstrokePointCount'].raw_value, 16)
            raw = rec['HandleForceCurve'].raw_value
            self.assertEqual(len(raw), 16)
            # First five samples match source; remainder padded with zeros
            self.assertEqual(raw[:5], (1000, 2000, 4000, 3000, 500))
            self.assertEqual(raw[5:], (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    def test_downsampled_point_count_32_on_longer_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_curve = tuple(range(10, 90, 5))  # 16 values: 10..85
            df = _synthetic_stroke_df(curve_values=long_curve)
            fit_path = os.path.join(tmp, 'ds32.fit')
            fitwrite.write_fit(
                fit_path,
                df,
                row_date='2026-08-03',
                sport='rowing',
                instroke_export='downsampled',
                instroke_downsample_points=32,
                overwrite=True,
            )
            rec = _record_field_maps(fit_path)[0]
            self.assertEqual(rec['InstrokePointCount'].raw_value, 32)
            self.assertEqual(len(rec['HandleForceCurve'].raw_value), 32)

    def test_downsampled_interval_updates_with_point_count(self):
        """Interval mm = drive_mm / (point_count - 1) when using distance abscissa."""
        with tempfile.TemporaryDirectory() as tmp:
            fit_path, _ = _export_synthetic(
                tmp, 'downsampled', instroke_downsample_points=16,
            )
            rec = _record_field_maps(fit_path)[0]
            # 1000 mm / 15 ≈ 67
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

    def test_quiske_multi_curve_field_ids_and_scales(self):
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
            self.assertEqual(fmap[60]['field_name'], 'BoatAcceleratorCurve')
            self.assertEqual(fmap[60]['scale'], 100)
            self.assertEqual(fmap[60]['units'], 'm/s^2')
            self.assertEqual(fmap[61]['field_name'], 'OarAngleVelocityCurve')
            self.assertEqual(fmap[61]['scale'], 10)
            self.assertEqual(fmap[61]['units'], 'deg/s')
            rec = _record_field_maps(outfile)[0]
            self.assertIn('BoatAcceleratorCurve', rec)
            self.assertIn('OarAngleVelocityCurve', rec)
            self.assertEqual(len(rec['BoatAcceleratorCurve'].raw_value), 16)
            self.assertEqual(len(rec['OarAngleVelocityCurve'].raw_value), 16)

    def test_quiske_boat_accelerator_encoded_values_clipped_nonnegative(self):
        """UINT16 curves clip negatives to zero; encoded = physical * Y scale (100)."""
        import rowingdata
        csvfile = 'testdata/quiske_per_stroke_left.csv'
        with tempfile.TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, 'quiske_accel.fit')
            r = rowingdata.QuiskeParser(csvfile)
            row = rowingdata.rowingdata(df=r.df, absolutetimestamps=False)
            row.exporttofit(
                outfile,
                sport='rowing',
                instroke_export='downsampled',
                instroke_downsample_points=16,
            )
            curves, _ = fitwrite._get_instroke_curve_for_export(
                r.df, 'boat accelerator curve', 'downsampled', 16,
            )
            exported = curves[0]
            expected = tuple(
                int(max(0, round(float(v) * 100))) for v in exported
            )
            rec = _record_field_maps(outfile)[0]
            self.assertEqual(rec['BoatAcceleratorCurve'].raw_value, expected)

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

    def test_fitparser_quiske_curve_columns_roundtrip(self):
        """FITParser decodes exported downsampled curves to the same physical values as fitparse."""
        import rowingdata
        csvfile = 'testdata/quiske_per_stroke_left.csv'
        with tempfile.TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, 'quiske_roundtrip.fit')
            r = rowingdata.QuiskeParser(csvfile)
            row = rowingdata.rowingdata(df=r.df, absolutetimestamps=False)
            row.exporttofit(
                outfile,
                sport='rowing',
                instroke_export='downsampled',
                instroke_downsample_points=16,
            )
            rec = _record_field_maps(outfile)[0]
            rr = rowingdata.FITParser(outfile)
            self.assertIn('boat accelerator curve', rr.df.columns)
            self.assertIn('oar angle velocity curve', rr.df.columns)
            expected_oar = [
                v / 10.0 for v in rec['OarAngleVelocityCurve'].raw_value
            ]
            out_oar = _parse_curve_parenthesized(
                rr.df['oar angle velocity curve'].iloc[0],
            )
            self.assertEqual(out_oar, expected_oar)
            expected_accel = [
                v / 100.0 for v in rec['BoatAcceleratorCurve'].raw_value
            ]
            out_accel = _parse_curve_parenthesized(
                rr.df['boat accelerator curve'].iloc[0],
            )
            self.assertEqual(out_accel, expected_accel)


if __name__ == '__main__':
    unittest.main()
