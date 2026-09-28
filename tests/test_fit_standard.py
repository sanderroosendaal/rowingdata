"""FIT export conformance with the Rowing Data Standard draft.

Source of truth: https://github.com/MoveLab-Studio/rowing-data-standard
(Draft v0.1, pinned commit 5ef01be6de48cbea7d5c57213f778e2fa63353df).

These tests assert raw encoded integers, not round-trips through our own
parser, so a scale or unit change upstream fails the build here.
"""
from __future__ import absolute_import

import os
import shutil
import sys
import tempfile
import unittest
from uuid import NAMESPACE_DNS, uuid5

import pandas as pd
from fitparse import FitFile

import rowingdata
from rowingdata import fitwrite

GOLDEN_FIT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'testdata',
    'rowingdata_standard_example.fit'
)


def _sample_df(dual_oarlock=False, curve=False):
    data = {
        'TimeStamp (sec)': [1.0e9, 1.0e9 + 2.1, 1.0e9 + 4.2],
        ' Horizontal (meters)': [0.0, 8.5, 17.0],
        'cum_dist': [0.0, 8.5, 17.0],
        ' Cadence (stokes/min)': [28.5, 29.0, 28.25],
        ' HRCur (bpm)': [140, 141, 142],
        ' Power (watts)': [200, 210, 205],
        ' Stroke500mPace (sec/500m)': [120.0, 118.0, 119.0],
        ' DriveLength (meters)': [1.42, 1.43, 1.41],
        ' StrokeDistance (meters)': [8.5, 8.5, 8.5],
        ' DriveTime (ms)': [800, 810, 790],
        ' StrokeRecoveryTime (ms)': [1300, 1260, 1350],
        ' AverageDriveForce (N)': [412.3, 400.0, 410.0],
        ' PeakDriveForce (N)': [800.0, 790.0, 805.0],
        ' AverageBoatSpeed (m/s)': [4.17, 4.20, 4.10],
        ' WorkoutState': [1, 1, 1],
        ' WorkPerStroke (joules)': [250, 255, 248],
        ' DragFactor': [110, 110, 110],
        ' Stroke Number': [1, 2, 3],
        'catch': [-32.1, -31.8, -32.4],
        'finish': [20.5, 20.1, 20.8],
        'effectiveLength': [0.87, 0.88, 0.86],
    }
    if dual_oarlock:
        data.update({
            'catch_port': [-32.0, -31.5, -32.2],
            'catch_starboard': [-32.2, -32.1, -32.6],
            'effectiveLength_port': [0.87, 0.88, 0.86],
            'effectiveLength_starboard': [0.88, 0.89, 0.87],
        })
    if curve:
        data['curve_data'] = ['[10,40,80,60,20]', '[12,44,84,62,22]', '[11,42,82,61,21]']
    return pd.DataFrame(data)


def _app_ids(fit):
    ids = []
    for msg in fit.get_messages('developer_data_id'):
        value = msg.get_value('application_id')
        if not isinstance(value, bytes):
            value = bytes(bytearray(value))
        ids.append(value)
    return ids


def _dev_index_by_app_id(fit):
    mapping = {}
    for msg in fit.get_messages('developer_data_id'):
        value = msg.get_value('application_id')
        if not isinstance(value, bytes):
            value = bytes(bytearray(value))
        mapping[msg.get_value('developer_data_index')] = value
    return mapping


def _descriptions(fit):
    out = {}
    for msg in fit.get_messages('field_description'):
        name = msg.get_value('field_name')
        if isinstance(name, list):
            name = name[0]
        out[msg.get_value('field_definition_number')] = {
            'name': name,
            'scale': msg.get_value('scale'),
            'units': msg.get_value('units'),
            'dev_index': msg.get_value('developer_data_index'),
        }
    return out


def _record_field(record, name):
    for field in record:
        if field.name == name:
            return field
    return None


class TestStandardEncoding(unittest.TestCase):
    """Raw encodings must match the draft field tables."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.outfile = os.path.join(self.tmpdir, 'standard.fit')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, df=None, **kwargs):
        fitwrite.write_fit(
            self.outfile, _sample_df() if df is None else df,
            row_date='2016-01-01', **kwargs
        )
        return FitFile(self.outfile, check_crc=False)

    def test_application_id_is_standard_uuid(self):
        fit = self._write()
        expected = uuid5(NAMESPACE_DNS, 'rowingdata').bytes
        assert fitwrite.ROWINGDATA_APP_ID == expected
        assert expected in _app_ids(fit)

    def test_drive_length_is_millimetres(self):
        fit = self._write()
        desc = _descriptions(fit)[0]
        assert desc['name'] == 'DriveLength'
        assert desc['units'] == 'mm'
        assert int(desc['scale']) == 1
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'DriveLength').raw_value == 1420

    def test_average_boat_speed_uses_scale_255(self):
        fit = self._write()
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'AverageBoatSpeed').raw_value == int(round(4.17 * 255))
        # Scale 255 is the uint8 invalid value, so it cannot travel in the
        # field_description scale byte. Consumers take it from the registry.
        assert _descriptions(fit)[8]['scale'] is None

    def test_session_carries_recording_strategy(self):
        fit = self._write()
        session = list(fit.get_messages('session'))[0]
        strategy = _record_field(session, 'RecordingStrategy')
        assert strategy is not None
        assert int(strategy.value) == fitwrite.RECORDING_STRATEGY_STROKE_BOUNDARY

    def test_stroke_rate_with_native_cadence(self):
        fit = self._write()
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'StrokeRate').raw_value == 2850
        assert record.get_value('cadence') == 28
        assert _record_field(record, 'fractional_cadence').raw_value == 64

    def test_stroke_work_and_forces(self):
        fit = self._write()
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'StrokeWork').raw_value == 250
        assert _record_field(record, 'AverageDriveForceN').raw_value == 4123

    def test_effective_length_is_millimetres(self):
        fit = self._write()
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'EffectiveLength').raw_value == 870
        assert _descriptions(fit)[16]['units'] == 'mm'

    def test_dual_effective_length_is_not_emitted(self):
        """Registry flags 210/211 as unresolved (mm vs m); do not write them."""
        fit = self._write(df=_sample_df(dual_oarlock=True))
        desc = _descriptions(fit)
        assert 200 in desc and desc[200]['name'] == 'CatchPort'
        assert 201 in desc
        assert 210 not in desc
        assert 211 not in desc
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'EffectiveLengthPort') is None
        assert _record_field(record, 'EffectiveLengthStarboard') is None


class TestInstrokeNamespace(unittest.TestCase):
    """Unallocated curve IDs must not sit under the standard application ID."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_curves(self, mode):
        path = os.path.join(self.tmpdir, 'instroke_%s.fit' % mode)
        fitwrite.write_fit(
            path, _sample_df(curve=True), row_date='2016-01-01',
            instroke_export=mode, instroke_columns=['curve_data']
        )
        return FitFile(path, check_crc=False)

    def test_private_and_standard_namespaces_differ(self):
        assert fitwrite.INSTROKE_APP_ID != fitwrite.ROWINGDATA_APP_ID

    def test_curve_arrays_use_private_application_id(self):
        fit = self._write_curves('full')
        apps = _app_ids(fit)
        assert fitwrite.ROWINGDATA_APP_ID in apps
        assert fitwrite.INSTROKE_APP_ID in apps
        by_index = _dev_index_by_app_id(fit)
        desc = _descriptions(fit)
        assert by_index[desc[60]['dev_index']] == fitwrite.INSTROKE_APP_ID
        # Axis metadata 90-92 is assigned in the registry, so it stays standard.
        for field_id in (90, 91, 92):
            assert by_index[desc[field_id]['dev_index']] == fitwrite.ROWINGDATA_APP_ID

    def test_summary_fields_use_private_application_id(self):
        fit = self._write_curves('summary')
        by_index = _dev_index_by_app_id(fit)
        desc = _descriptions(fit)
        summary_ids = [fid for fid, d in desc.items() if 'HandleForceCurve_' in str(d['name'])]
        assert summary_ids
        for fid in summary_ids:
            assert by_index[desc[fid]['dev_index']] == fitwrite.INSTROKE_APP_ID

    def test_core_fields_stay_on_standard_id_with_curves(self):
        fit = self._write_curves('full')
        by_index = _dev_index_by_app_id(fit)
        desc = _descriptions(fit)
        for field_id in (0, 9, 10, 93):
            assert by_index[desc[field_id]['dev_index']] == fitwrite.ROWINGDATA_APP_ID


class TestReaderMapsStandardFields(unittest.TestCase):
    """FITParser turns standard developer fields back into CSV columns."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.outfile = os.path.join(self.tmpdir, 'roundtrip.fit')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip_physical_units(self):
        fitwrite.write_fit(self.outfile, _sample_df(), row_date='2016-01-01')
        df = rowingdata.FITParser(self.outfile).df
        assert abs(float(df[' DriveLength (meters)'].iloc[0]) - 1.42) < 0.005
        assert abs(float(df[' AverageBoatSpeed (m/s)'].iloc[0]) - 4.17) < 0.01
        assert abs(float(df[' AverageDriveForce (N)'].iloc[0]) - 412.3) < 0.1
        assert abs(float(df['effectiveLength'].iloc[0]) - 0.87) < 0.005
        assert abs(float(df['catch'].iloc[0]) - (-32.1)) < 0.05
        assert int(df[' DriveTime (ms)'].iloc[0]) == 800
        assert int(df[' WorkPerStroke (joules)'].iloc[0]) == 250
        assert abs(float(df[' Cadence (stokes/min)'].iloc[0]) - 28.5) < 0.02

    def test_legacy_fit_file_still_parses(self):
        df = rowingdata.FITParser('testdata/3x250m.fit').df
        assert len(df) > 0
        assert df[' Horizontal (meters)'].max() > 0


class TestGoldenFile(unittest.TestCase):
    """The committed example the standard repo reads in its interop test."""

    def test_golden_file_is_committed_and_conformant(self):
        assert os.path.isfile(GOLDEN_FIT)
        fit = FitFile(GOLDEN_FIT, check_crc=False)
        assert fitwrite.ROWINGDATA_APP_ID in _app_ids(fit)
        desc = _descriptions(fit)
        assert desc[0]['units'] == 'mm'
        assert len(list(fit.get_messages('record'))) > 0

    def test_golden_file_parses_into_csv_columns(self):
        df = rowingdata.FITParser(GOLDEN_FIT).df
        assert float(df[' DriveLength (meters)'].max()) < 5.0


class TestStandardPackageInterop(unittest.TestCase):
    """Optional: read our export with the committee's sample implementation."""

    @staticmethod
    def _standard_src():
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, '..', '..', 'rowing-data-standard', 'python', 'src'),
            os.path.join(here, '..', 'rowing-data-standard', 'python', 'src'),
        ]
        return next((p for p in candidates if os.path.isdir(p)), None)

    def test_rowing_data_reads_our_export(self):
        src = self._standard_src()
        if src is None:
            raise unittest.SkipTest('rowing-data-standard checkout not found')
        if src not in sys.path:
            sys.path.insert(0, src)
        try:
            from rowing_data import APPLICATION_ID, read_fit, validate
        except ImportError:
            raise unittest.SkipTest('rowing_data package not importable')
        assert APPLICATION_ID == fitwrite.ROWINGDATA_APP_ID
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'interop.fit')
            fitwrite.write_fit(path, _sample_df(), row_date='2016-01-01')
            session = read_fit(path)
            assert len(session.records) == 3
            assert session.records[0].drive_length_mm == 1420
            errors = [
                issue for issue in validate(session)
                if getattr(issue, 'level', '') == 'error'
            ]
            assert errors == []
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
