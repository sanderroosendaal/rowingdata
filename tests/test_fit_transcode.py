# -*- coding: utf-8 -*-
"""Tests for Garmin FIT transcode and Split message preservation."""
from __future__ import absolute_import

import os
import tempfile
import unittest

import pytest

from rowingdata.fit_garmin_bridge import (
    MESG_SPLIT,
    MESG_SPLIT_SUMMARY,
    iter_preserved_generic_messages,
)

# Default sample FIT file with splits for testing
_DEFAULT_SAMPLE_FIT = os.path.join(
    os.path.dirname(__file__),
    '..',
    'testdata',
    'sample_with_splits.fit'
)


def test_mesg_constants_document_garmin_extended_profile():
    """312/313 are Split / SplitSummary in current Garmin FIT profiles (not in older fitparse names)."""
    assert MESG_SPLIT == 312
    assert MESG_SPLIT_SUMMARY == 313


def test_preserved_message_roundtrip_from_sample():
    """Test that preserved messages can be read from a FIT file with splits."""
    # Use custom sample if set, otherwise use default
    path = os.environ.get('ROWINGDATA_GARMIN_SAMPLE_FIT')
    if not path or not os.path.isfile(path):
        path = _DEFAULT_SAMPLE_FIT
    
    n = sum(1 for _ in iter_preserved_generic_messages(path))
    assert n >= 3  # at least 3 split messages (minimal sample)


def test_transcode_writes_split_messages():
    """Test that transcoding preserves Split messages from source FIT."""
    from collections import Counter
    from fitparse import FitFile

    from rowingdata.fit_transcode import data_frame_from_garmin_fit
    from rowingdata.fitwrite import write_fit

    # Use custom sample if set, otherwise use default
    src = os.environ.get('ROWINGDATA_GARMIN_SAMPLE_FIT')
    if not src or not os.path.isfile(src):
        src = _DEFAULT_SAMPLE_FIT
    
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
        # Default sample has split messages, but no workout_step
        assert c.get('unknown_312', 0) >= 1
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
