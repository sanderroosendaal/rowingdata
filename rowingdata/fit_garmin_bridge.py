# -*- coding: utf-8 -*-
"""
Bridge Garmin / OpenRowingMonitor FIT messages (Split, SplitSummary, Workout, WorkoutStep)
from fitparse into fit_tool GenericMessage for re-emission with rowingdata exports.

Global message numbers 312 and 313 are not in fitparse's profile (shown as unknown_312 / unknown_313);
they correspond to Split and SplitSummary in current Garmin FIT profiles.
"""
from __future__ import absolute_import

from fit_tool.base_type import BaseType
from fit_tool.definition_message import DefinitionMessage
from fit_tool.endian import Endian
from fit_tool.field import Field
from fit_tool.field_definition import FieldDefinition
from fit_tool.generic_message import GenericMessage

from fitparse.base import FitFile as FitparseFitFile
# Garmin FIT profile (extended): Split / SplitSummary message numbers
MESG_SPLIT = 312
MESG_SPLIT_SUMMARY = 313
MESG_WORKOUT = 26
MESG_WORKOUT_STEP = 27

_DEFAULT_PRESERVE = (MESG_WORKOUT, MESG_WORKOUT_STEP, MESG_SPLIT_SUMMARY, MESG_SPLIT)


def _fitparse_base_type_to_fit_tool(fp_bt):
    """Map fitparse records.BaseType name to fit_tool BaseType."""
    name = fp_bt.name.lower()
    mapping = {
        'enum': BaseType.ENUM,
        'sint8': BaseType.SINT8,
        'uint8': BaseType.UINT8,
        'string': BaseType.STRING,
        'sint16': BaseType.SINT16,
        'uint16': BaseType.UINT16,
        'sint32': BaseType.SINT32,
        'uint32': BaseType.UINT32,
        'float32': BaseType.FLOAT32,
        'float64': BaseType.FLOAT64,
        'byte': BaseType.BYTE,
        'uint8z': BaseType.UINT8Z,
        'uint16z': BaseType.UINT16Z,
        'uint32z': BaseType.UINT32Z,
        'uint64': BaseType.UINT64,
        'sint64': BaseType.SINT64,
    }
    return mapping.get(name, BaseType.UINT8)


def _field_definition_from_fitparse(fd):
    bt = _fitparse_base_type_to_fit_tool(fd.base_type)
    return FieldDefinition(field_id=fd.def_num, size=fd.size, base_type=bt)


def _set_field_from_raw(fit_field, raw_value):
    """Copy raw decoded value from fitparse into fit_tool Field.encoded_values."""
    bt = fit_field.base_type
    if raw_value is None:
        inv = bt.invalid_raw_value() if hasattr(bt, 'invalid_raw_value') else 0
        fit_field.set_encoded_value(0, inv, check_validity=False)
        return
    if isinstance(raw_value, tuple):
        for i, rv in enumerate(raw_value):
            fit_field.set_encoded_value(i, int(rv) & 0xFFFFFFFF, check_validity=False)
        return
    if bt == BaseType.STRING:
        fit_field.growable = True
        s = raw_value if isinstance(raw_value, str) else str(raw_value)
        fit_field.set_value(0, s[: fit_field.size] if fit_field.size else s)
        return
    n = len(fit_field.encoded_values)
    if n == 1:
        fit_field.set_encoded_value(0, int(raw_value), check_validity=False)
    else:
        fit_field.set_encoded_value(0, int(raw_value), check_validity=False)


def _generic_from_fitparse_data(def_mesg, data_msg, local_id):
    """Build fit_tool DefinitionMessage + GenericMessage from fitparse messages."""
    endian = Endian.LITTLE if def_mesg.endian == '<' else Endian.BIG
    field_defs = [_field_definition_from_fitparse(fd) for fd in def_mesg.field_defs]
    definition = DefinitionMessage(
        local_id=local_id,
        global_id=def_mesg.mesg_num,
        endian=endian,
        field_definitions=field_defs,
        developer_field_definitions=[],
    )
    gen = GenericMessage(definition_message=definition, developer_fields=None)
    # Zip field data: fitparse DataMessage.fields order matches def field_defs + dev
    fp_fields = list(data_msg.fields)
    for i, fd in enumerate(def_mesg.field_defs):
        if i >= len(fp_fields):
            break
        fld = gen.get_field(fd.def_num)
        if fld is None and i < len(gen.fields):
            fld = gen.fields[i]
        if fld is None:
            continue
        fv = fp_fields[i]
        raw = fv.raw_value if fv.raw_value is not None else fv.value
        _set_field_from_raw(fld, raw)
    return definition, gen


def iter_preserved_generic_messages(fit_path, mesg_nums=None):
    """
    Parse FIT with fitparse low-level API; yield (definition, GenericMessage) pairs
    for each data message whose global mesg_num is in mesg_nums, in file order.
    Local message IDs are reassigned uniquely for fit_tool (starting at 8 to avoid
    clashing with common record/lap local IDs).
    """
    if mesg_nums is None:
        mesg_nums = _DEFAULT_PRESERVE
    mesg_set = set(mesg_nums)
    f = FitparseFitFile(fit_path)
    while f._parse_message():
        pass
    # Map (mesg_num, tuple of field def signatures) -> next local_id
    # Use local_id >= 8 to avoid clashing with fit_tool defaults (record/lap/event often use 0–3).
    def_key_to_local = {}
    next_local = 8
    for m in f._messages:
        if getattr(m, 'type', None) != 'data':
            continue
        if m.mesg_num not in mesg_set:
            continue
        def_mesg = m.def_mesg
        key = (m.mesg_num, tuple((fd.def_num, fd.size, fd.base_type.name) for fd in def_mesg.field_defs))
        if key not in def_key_to_local:
            def_key_to_local[key] = next_local
            next_local += 1
        local_id = def_key_to_local[key]
        definition, gen = _generic_from_fitparse_data(def_mesg, m, local_id)
        yield definition, gen


def add_preserved_messages_to_builder(builder, fit_path, mesg_nums=None):
    """
    Add preserved native messages to a FitFileBuilder.

    Emits each DefinitionMessage once (required by FIT), then each data message with
    ``definition_message`` cleared so FitFileBuilder binds the message to the stored
    definition (see fit_tool FitFileBuilder.add).
    """
    seen_def = set()
    for definition, gen in iter_preserved_generic_messages(fit_path, mesg_nums=mesg_nums):
        key = (definition.global_id, definition.local_id, tuple((fd.field_id, fd.size, fd.base_type) for fd in definition.field_definitions))
        if key not in seen_def:
            builder.add(definition)
            seen_def.add(key)
        gen.definition_message = None
        builder.add(gen)
