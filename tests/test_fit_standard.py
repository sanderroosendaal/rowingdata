"""FIT export conformance with the Rowing Data Standard draft.

Source of truth: https://github.com/Rowing-Industry-Trade-Association/rowing-data-standard
(pinned commit fdf2d3b6503f433969d31f4c16201fcd8b3edc53).

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
        'effectiveArc': [85.0, 86.5, 84.0],
    }
    if dual_oarlock:
        data.update({
            'catch_port': [-32.0, -31.5, -32.2],
            'catch_starboard': [-32.2, -32.1, -32.6],
            'effectiveArc_port': [85.0, 86.5, 84.0],
            'effectiveArc_starboard': [85.5, 87.0, 84.5],
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

    def test_average_boat_speed_uses_scale_100(self):
        fit = self._write()
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'AverageBoatSpeed').raw_value == 417
        assert int(_descriptions(fit)[8]['scale']) == 100

    def test_session_carries_recording_strategy(self):
        fit = self._write()
        session = list(fit.get_messages('session'))[0]
        strategy = _record_field(session, 'RecordingStrategy')
        assert strategy is not None
        assert int(strategy.value) == fitwrite.RECORDING_STRATEGY_STROKE_BOUNDARY

    def test_stroke_rate_with_native_cadence(self):
        fit = self._write()
        record = list(fit.get_messages('record'))[0]
        # 28.5 spm: cadence rounds half up, cadence256 carries the fraction
        assert record.get_value('cadence') == 29
        assert _record_field(record, 'cadence256').raw_value == 7296
        assert _record_field(record, 'fractional_cadence') is None
        assert _record_field(record, 'StrokeRate') is None
        assert 93 not in _descriptions(fit)

    def test_protocol_version_is_written(self):
        fit = self._write()
        versions = [
            msg.get_value('application_version')
            for msg in fit.get_messages('developer_data_id')
        ]
        assert versions == [1]

    def test_workout_state_field_9_is_never_written(self):
        """Field 9 was removed from the standard; intensity lives on the Lap only."""
        df = _sample_df()
        df[' WorkoutState'] = [1, 3, 5]  # rowing, interval rest, interval work
        fit = self._write(df=df)
        assert 9 not in _descriptions(fit)
        for r in fit.get_messages('record'):
            assert _record_field(r, 'WorkoutState') is None
        laps = list(fit.get_messages('lap'))
        assert [lap.get_value('intensity') for lap in laps] == ['active', 'rest', 5]  # 5 = interval

    def test_unknown_abscissa_curve_is_not_written(self):
        """InstrokeAbscissaType 0 must never be written (§6.3): the curve is left out."""
        import warnings
        df = _sample_df(curve=True).drop(columns=[' DriveTime (ms)', ' DriveLength (meters)'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit = self._write(df=df, instroke_export='downsampled')
        assert any('X-axis' in str(w.message) for w in caught)
        descriptions = _descriptions(fit)
        for field_id in (60, 90, 91, 92):
            assert field_id not in descriptions

    def test_abscissa_fields_only_with_curve_data(self):
        df = _sample_df(curve=True)
        df.loc[1, 'curve_data'] = None
        fit = self._write(df=df, instroke_export='downsampled')
        for r, has_curve in zip(fit.get_messages('record'), (True, False, True)):
            assert (_record_field(r, 'InstrokeAbscissaType') is not None) == has_curve
            if has_curve:
                assert _record_field(r, 'InstrokeAbscissaType').raw_value != 0

    def test_curve_without_known_interval_is_dropped_with_its_axis(self):
        """A Record whose sample interval is unknown carries no curve and no axis (§6.3)."""
        df = _sample_df(curve=True)
        df.loc[1, ' DriveLength (meters)'] = 0  # interval cannot be derived for this stroke
        fit = self._write(df=df, instroke_export='downsampled')
        for r, expected in zip(fit.get_messages('record'), (True, False, True)):
            names = {f.name for f in r}
            for name in ('HandleForceCurve', 'InstrokeAbscissaType',
                         'InstrokeSampleInterval', 'InstrokePointCount'):
                assert (name in names) == expected, name

    def test_dual_oarlock_summary_is_side_average(self):
        df = _sample_df(dual_oarlock=True)
        df['catch'] = [-10.0, -10.0, -10.0]  # source summary is overridden by the average
        fit = self._write(df=df)
        first = next(iter(fit.get_messages('record')))
        assert _record_field(first, 'Catch').raw_value == round(-32.1 * 10)  # (-32.0 + -32.2) / 2
        assert _record_field(first, 'EffectiveArc').raw_value == round(85.25 * 10)

    def test_every_lap_has_intensity_and_rest_is_its_own_lap(self):
        df = _sample_df()
        df[' WorkoutState'] = [1, 3, 1]
        fit = self._write(df=df)
        laps = list(fit.get_messages('lap'))
        assert [lap.get_value('intensity') for lap in laps] == ['active', 'rest', 'active']
        single = self._write()
        laps = list(single.get_messages('lap'))
        assert [lap.get_value('intensity') for lap in laps] == ['active']

    def test_time_sampled_session_and_stroke_state(self):
        df = _sample_df()
        df[' StrokeState'] = [2, 3, 4]
        fit = self._write(
            df=df, recording_strategy=fitwrite.RECORDING_STRATEGY_TIME_SAMPLED)
        session = list(fit.get_messages('session'))[0]
        assert int(_record_field(session, 'RecordingStrategy').value) == 3
        states = [
            _record_field(r, 'StrokeState').raw_value for r in fit.get_messages('record')
        ]
        assert states == [2, 3, 4]

    def test_stroke_state_only_with_time_sampled(self):
        df = _sample_df()
        df[' StrokeState'] = [2, 3, 4]
        fit = self._write(df=df)
        assert 96 not in _descriptions(fit)

    def test_oarlock_thresholds_on_session(self):
        fit = self._write(slip_threshold=80, wash_threshold=60)
        session = list(fit.get_messages('session'))[0]
        assert _record_field(session, 'SlipThreshold').raw_value == 80
        assert _record_field(session, 'WashThreshold').raw_value == 60
        desc = _descriptions(fit)
        assert desc[94]['units'] == 'N' and desc[95]['units'] == 'N'

    def test_lbs_columns_are_written_as_newtons(self):
        df = _sample_df().drop(columns=[' AverageDriveForce (N)', ' PeakDriveForce (N)'])
        df[' AverageDriveForce (lbs)'] = [100.0, 100.0, 100.0]
        df[' PeakDriveForce (lbs)'] = [200.0, 200.0, 200.0]
        fit = self._write(df=df)
        desc = _descriptions(fit)
        assert 4 not in desc and 5 not in desc
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'AverageDriveForce').raw_value == int(round(444.82216 * 10))

    def test_stroke_work_and_forces(self):
        fit = self._write()
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'StrokeWork').raw_value == 250
        assert _record_field(record, 'AverageDriveForce').raw_value == 4123

    def test_effective_arc_is_degrees(self):
        fit = self._write()
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'EffectiveArc').raw_value == 850
        assert int(_descriptions(fit)[16]['scale']) == 10
        assert _descriptions(fit)[16]['units'] == 'deg'
        assert _descriptions(fit)[16]['name'] == 'EffectiveArc'

    def test_dual_effective_arc_is_emitted(self):
        fit = self._write(df=_sample_df(dual_oarlock=True))
        desc = _descriptions(fit)
        assert 200 in desc and desc[200]['name'] == 'CatchPort'
        assert desc[210]['name'] == 'EffectiveArcPort'
        assert desc[211]['name'] == 'EffectiveArcStarboard'
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'EffectiveArcPort').raw_value == 850
        assert _record_field(record, 'EffectiveArcStarboard').raw_value == 855

    def test_invalid_values_are_never_written_as_measurements(self):
        """§6.5: 255 / 65535 are FIT invalid values, so a measurement clips below them."""
        df = _sample_df()
        df[' AverageDriveForce (N)'] = [1.0e6, 400.0, 410.0]
        df[' HRCur (bpm)'] = [300, 141, 142]
        fit = self._write(df=df)
        record = list(fit.get_messages('record'))[0]
        assert _record_field(record, 'AverageDriveForce').raw_value == 65534
        assert record.get_value('heart_rate') == 254


class TestInstrokeNamespace(unittest.TestCase):
    """HandleForceCurve is the one curve in the FIT file; unallocated IDs stay private."""

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

    def test_handle_force_curve_uses_standard_application_id(self):
        fit = self._write_curves('full')
        by_index = _dev_index_by_app_id(fit)
        desc = _descriptions(fit)
        assert desc[60]['name'] == 'HandleForceCurve'
        assert fitwrite.INSTROKE_APP_ID not in _app_ids(fit)
        for field_id in (60, 90, 91, 92):
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
        for field_id in (0, 10):
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
        assert abs(float(df['effectiveArc'].iloc[0]) - 85.0) < 0.05
        assert abs(float(df['catch'].iloc[0]) - (-32.1)) < 0.05
        assert int(df[' DriveTime (ms)'].iloc[0]) == 800
        assert int(df[' WorkPerStroke (joules)'].iloc[0]) == 250
        assert abs(float(df[' Cadence (stokes/min)'].iloc[0]) - 28.5) < 0.02

    def test_cadence256_is_read_as_stroke_rate(self):
        df = _sample_df()
        df[' Cadence (stokes/min)'] = [28.4, 29.0, 28.25]
        fitwrite.write_fit(self.outfile, df, row_date='2016-01-01')
        parsed = rowingdata.FITParser(self.outfile).df
        assert abs(float(parsed[' Cadence (stokes/min)'].iloc[0]) - 28.4) < 0.01
        assert abs(float(parsed[' Cadence (stokes/min)'].iloc[2]) - 28.25) < 0.01

    def test_workout_state_round_trips_through_intensity(self):
        df = _sample_df()
        df[' WorkoutState'] = [1, 3, 5]
        fitwrite.write_fit(self.outfile, df, row_date='2016-01-01')
        parsed = rowingdata.FITParser(self.outfile).df
        assert list(parsed[' WorkoutState']) == [1, 3, 5]

    def _write_file_with_field_9(self, states, protocol_version):
        """An older file: record field 9 (WorkoutState), written by hand since
        the exporter no longer produces it. No Lap carries an intensity, so the
        record field is the only source of the state."""
        from fit_tool.base_type import BaseType
        from fit_tool.developer_field import DeveloperField
        from fit_tool.fit_file_builder import FitFileBuilder
        from fit_tool.profile.messages.developer_data_id_message import DeveloperDataIdMessage
        from fit_tool.profile.messages.field_description_message import FieldDescriptionMessage
        from fit_tool.profile.messages.file_id_message import FileIdMessage
        from fit_tool.profile.messages.record_message import RecordMessage
        from fit_tool.profile.profile_type import FileType, Manufacturer

        builder = FitFileBuilder(auto_define=True, min_string_size=50)
        file_id = FileIdMessage()
        file_id.type = FileType.ACTIVITY
        file_id.manufacturer = Manufacturer.GARMIN.value
        file_id.time_created = 1_000_000_000_000
        builder.add(file_id)
        dev_id = DeveloperDataIdMessage()
        dev_id.application_id = fitwrite.ROWINGDATA_APP_ID
        dev_id.developer_data_index = 0
        if protocol_version is not None:
            dev_id.application_version = protocol_version
        builder.add(dev_id)
        desc = FieldDescriptionMessage()
        desc.developer_data_index = 0
        desc.field_definition_number = 9
        desc.fit_base_type_id = BaseType.UINT8.value
        desc.field_name = 'WorkoutState'
        desc.scale = 1
        desc.offset = 0
        desc.units = ''
        builder.add(desc)
        for i, state in enumerate(states):
            field = DeveloperField(
                developer_data_index=0, field_id=9, size=1, name='WorkoutState',
                base_type=BaseType.UINT8, scale=1, offset=0, units='')
            field.set_value(0, state)
            rec = RecordMessage(developer_fields=[field])
            rec.timestamp = 1_000_000_000_000 + i * 2000
            rec.distance = 8.5 * (i + 1)
            builder.add(rec)
        builder.build().to_file(self.outfile)

    def test_field_9_is_still_read_from_older_files(self):
        """Field 9 left the standard, but files that carry it are still read."""
        # Protocol version 1: field 9 holds FIT intensity values (0 Active, 1 Rest, 5 Interval)
        self._write_file_with_field_9([0, 1, 5], protocol_version=1)
        parsed = rowingdata.FITParser(self.outfile).df
        assert list(parsed[' WorkoutState']) == [1, 3, 5]  # rowingdata codes
        # No application_version: pre-protocol files carry rowingdata codes as they are
        self._write_file_with_field_9([1, 3, 5], protocol_version=None)
        parsed = rowingdata.FITParser(self.outfile).df
        assert list(parsed[' WorkoutState']) == [1, 3, 5]

    def test_unknown_intensity_is_read_as_other(self):
        from rowingdata import otherparsers
        f = otherparsers._fit_intensity_to_workout_state
        assert f(0) == 1 and f('active') == 1
        assert f(1) == 3 and f('rest') == 3
        assert f(5) == 5 and f('interval') == 5
        assert f(200) == f(6)
        assert f(None) is None

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


class TestReaderKeepsRenamedFields(unittest.TestCase):
    """Files written against earlier drafts still read; the names are never written."""

    def test_old_names_map_to_current_columns(self):
        from rowingdata.otherparsers import _developer_field_spec
        spec = _developer_field_spec()
        assert spec['averagedriveforcen']['column'] == ' AverageDriveForce (N)'
        assert spec['peakdriveforcen']['column'] == ' PeakDriveForce (N)'
        assert spec['effectivearc']['column'] == 'effectiveArc'
        # EffectiveLength (mm) is not an arc, so it keeps its own column
        assert spec['effectivelength']['column'] == 'effectiveLength'
        assert spec['effectivelengthport']['column'] == 'effectiveLength_port'

    def test_invalid_value_is_read_as_missing(self):
        from rowingdata.otherparsers import _developer_field_spec
        assert _developer_field_spec()['averageboatspeed']['invalid'] == 65535
