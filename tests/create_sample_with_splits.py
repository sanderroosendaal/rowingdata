#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a minimal FIT file with Split messages for testing.
This file is committed to testdata/ for CI/Linux testing.
"""
from __future__ import absolute_import
import os
from datetime import datetime

from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.profile_type import Sport, FileType, Manufacturer, Event, EventType
from fit_tool.generic_message import GenericMessage
from fit_tool.definition_message import DefinitionMessage
from fit_tool.field_definition import FieldDefinition
from fit_tool.base_type import BaseType
from fit_tool.endian import Endian


def create_split_definition():
    """Create definition message for Split (mesg 312)."""
    field_defs = [
        FieldDefinition(field_id=253, size=4, base_type=BaseType.UINT32),  # timestamp
        FieldDefinition(field_id=0, size=1, base_type=BaseType.ENUM),       # split_type
        FieldDefinition(field_id=1, size=4, base_type=BaseType.UINT32),     # total_elapsed_time
        FieldDefinition(field_id=2, size=4, base_type=BaseType.UINT32),     # total_distance
        FieldDefinition(field_id=9, size=4, base_type=BaseType.UINT32),     # total_timer_time
    ]
    return DefinitionMessage(
        local_id=8,  # Use local_id 8+ to avoid clashing with standard messages
        global_id=312,  # Split message
        endian=Endian.LITTLE,
        field_definitions=field_defs,
        developer_field_definitions=[],
    )


def create_split_message(definition, timestamp_ms, split_type, elapsed, distance, timer_time):
    """Create a Split message (GenericMessage).
    
    timestamp_ms: Unix timestamp in milliseconds (will be converted to FIT epoch seconds)
    elapsed, timer_time: in milliseconds (1/1000s as per FIT spec)
    distance: in cm
    """
    gen = GenericMessage(definition_message=definition, developer_fields=None)
    # Timestamp: convert from Unix ms to FIT epoch seconds
    # FIT epoch is Dec 31, 1989; Unix epoch is Jan 1, 1970
    fit_offset_sec = 631065600  # seconds between Unix and FIT epochs
    timestamp_sec = (timestamp_ms // 1000) - fit_offset_sec
    
    # Set field values by field_id
    gen.get_field(253).set_value(0, timestamp_sec)  # timestamp (seconds since FIT epoch)
    gen.get_field(0).set_value(0, split_type)   # split_type (0 = interval)
    gen.get_field(1).set_value(0, elapsed)      # total_elapsed_time (1/1000s)
    gen.get_field(2).set_value(0, distance)     # total_distance (cm)
    gen.get_field(9).set_value(0, timer_time)   # total_timer_time (1/1000s)
    return gen


def create_sample_fit_with_splits(output_path):
    """Create a minimal FIT file with Split messages."""
    builder = FitFileBuilder(auto_define=True)
    
    # Timestamps: Unix timestamps in MILLISECONDS (fit_tool convention)
    # fit_tool converts internally to FIT epoch
    # Target date: Jan 1, 2024 10:00:00
    unix_epoch = datetime(1970, 1, 1, 0, 0, 0)
    target_date = datetime(2024, 1, 1, 10, 0, 0)
    base_ts_ms = int((target_date - unix_epoch).total_seconds() * 1000)
    
    # File ID
    file_id = FileIdMessage()
    file_id.type = FileType.ACTIVITY
    file_id.manufacturer = Manufacturer.DEVELOPMENT.value
    file_id.product = 0
    file_id.time_created = base_ts_ms
    file_id.serial_number = 12345
    builder.add(file_id)
    
    # Event (timer start)
    event_start = EventMessage()
    event_start.timestamp = base_ts_ms
    event_start.event = Event.TIMER
    event_start.event_type = EventType.START
    event_start.event_group = 0
    builder.add(event_start)
    
    # Create 3 intervals with records and splits
    interval_duration_ms = 120 * 1000  # 2 minutes per interval (milliseconds)
    interval_distance = 50000  # 500m per interval (in cm)
    
    for interval_idx in range(3):
        interval_start_ms = base_ts_ms + interval_idx * interval_duration_ms
        
        # Lap message for interval
        lap = LapMessage()
        lap.timestamp = interval_start_ms + interval_duration_ms
        lap.start_time = interval_start_ms
        lap.total_elapsed_time = 120  # seconds (fit_tool applies scale 1000)
        lap.total_timer_time = 120
        lap.total_distance = interval_distance / 100.0  # meters
        lap.total_strokes = 40  # ~20 spm
        lap.sport = Sport.ROWING
        builder.add(lap)
        
        # Add some record messages
        for stroke_idx in range(4):
            record = RecordMessage()
            record.timestamp = interval_start_ms + stroke_idx * 3000  # milliseconds
            record.distance = (interval_idx * interval_distance + stroke_idx * 12500) / 100.0  # meters
            record.cadence = 20  # spm
            record.heart_rate = 140 + stroke_idx
            builder.add(record)
    
    # Add Split messages (one per interval)
    split_def = create_split_definition()
    builder.add(split_def)
    
    for interval_idx in range(3):
        interval_end_ms = base_ts_ms + (interval_idx + 1) * interval_duration_ms
        cumulative_time = (interval_idx + 1) * 120 * 1000  # milliseconds
        cumulative_dist = (interval_idx + 1) * interval_distance  # cm
        
        split = create_split_message(
            split_def,
            timestamp_ms=interval_end_ms,
            split_type=0,  # interval
            elapsed=cumulative_time,
            distance=cumulative_dist,
            timer_time=cumulative_time,
        )
        split.definition_message = None  # Let builder reuse definition
        builder.add(split)
    
    # Session
    session = SessionMessage()
    session.timestamp = base_ts_ms + 3 * interval_duration_ms
    session.start_time = base_ts_ms
    session.total_elapsed_time = 3 * 120  # seconds (fit_tool applies scale 1000)
    session.total_timer_time = 3 * 120
    session.total_distance = 3 * interval_distance / 100.0  # meters
    session.sport = Sport.ROWING
    session.total_strokes = 120
    session.avg_cadence = 20
    builder.add(session)
    
    # Event (timer stop)
    event_stop = EventMessage()
    event_stop.timestamp = base_ts_ms + 3 * interval_duration_ms
    event_stop.event = Event.TIMER
    event_stop.event_type = EventType.STOP_ALL
    event_stop.event_group = 0
    builder.add(event_stop)
    
    # Activity
    activity = ActivityMessage()
    activity.timestamp = base_ts_ms + 3 * interval_duration_ms
    activity.total_timer_time = 3 * 120  # seconds (fit_tool applies scale 1000)
    activity.num_sessions = 1
    activity.type = 0  # manual
    activity.event = Event.ACTIVITY
    activity.event_type = EventType.STOP
    builder.add(activity)
    
    # Write to file
    fit_file = builder.build()
    fit_file.to_file(output_path)
    print(f"Created sample FIT with splits: {output_path}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    output = os.path.join(repo_root, 'testdata', 'sample_with_splits.fit')
    create_sample_fit_with_splits(output)
