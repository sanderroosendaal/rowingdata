# -*- coding: utf-8 -*-
"""Tests for Garmin FIT transcode and Split message preservation."""
from __future__ import absolute_import

import os
import tempfile

import pytest

from rowingdata.fit_garmin_bridge import (
    MESG_SPLIT,
    MESG_SPLIT_SUMMARY,
    iter_preserved_generic_messages,
)


def test_mesg_constants_document_garmin_extended_profile():
    """312/313 are Split / SplitSummary in current Garmin FIT profiles (not in older fitparse names)."""
    assert MESG_SPLIT == 312
    assert MESG_SPLIT_SUMMARY == 313


@pytest.mark.skipif(
    not os.environ.get('ROWINGDATA_GARMIN_SAMPLE_FIT')
    or not os.path.isfile(os.environ['ROWINGDATA_GARMIN_SAMPLE_FIT']),
    reason='Set ROWINGDATA_GARMIN_SAMPLE_FIT to a Garmin/ORM .fit with splits',
)
def test_preserved_message_roundtrip_from_sample():
    path = os.environ['ROWINGDATA_GARMIN_SAMPLE_FIT']
    n = sum(1 for _ in iter_preserved_generic_messages(path))
    assert n >= 5  # at least workout + steps + splits


@pytest.mark.skipif(
    not os.environ.get('ROWINGDATA_GARMIN_SAMPLE_FIT')
    or not os.path.isfile(os.environ['ROWINGDATA_GARMIN_SAMPLE_FIT']),
    reason='Set ROWINGDATA_GARMIN_SAMPLE_FIT to a Garmin/ORM .fit with splits',
)
def test_transcode_writes_split_messages():
    from collections import Counter
    from fitparse import FitFile

    from rowingdata.fit_transcode import data_frame_from_garmin_fit
    from rowingdata.fitwrite import write_fit

    src = os.environ['ROWINGDATA_GARMIN_SAMPLE_FIT']
    df = data_frame_from_garmin_fit(src)
    with tempfile.NamedTemporaryFile(suffix='.fit', delete=False) as tmp:
        out = tmp.name
    try:
        write_fit(
            out,
            df,
            row_date='2026-03-29',
            sport='indoor_rowing',
            use_developer_fields=True,
            instroke_export='full',
            overwrite=True,
            garmin_parity_source_fit=src,
        )
        f = FitFile(out)
        c = Counter(m.name for m in f.messages)
        assert c.get('unknown_312', 0) >= 1
        assert c.get('workout_step', 0) >= 1
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
