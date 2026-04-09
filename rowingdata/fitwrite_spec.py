"""
Load machine-readable FIT export spec (fit_export_spec.json) for fitwrite.
"""
from __future__ import absolute_import
from __future__ import print_function

import json
import os
import warnings

try:
    from importlib import resources as importlib_resources
except ImportError:
    import importlib_resources  # type: ignore

_BASE_TYPE_MAP = None


def _get_base_type_map():
    global _BASE_TYPE_MAP
    if _BASE_TYPE_MAP is not None:
        return _BASE_TYPE_MAP
    try:
        from fit_tool.base_type import BaseType
        _BASE_TYPE_MAP = {
            'UINT8': BaseType.UINT8,
            'UINT16': BaseType.UINT16,
            'SINT16': BaseType.SINT16,
        }
    except ImportError:
        _BASE_TYPE_MAP = None
    return _BASE_TYPE_MAP

SPEC_FILENAME = 'fit_export_spec.json'
_PACKAGE = 'rowingdata'
_DATA_SUBDIR = 'data'

_SPEC_CACHE = None


def _base_type_from_string(name):
    mapping = _get_base_type_map()
    if mapping is None:
        raise ImportError('fit-tool is required to resolve FIT BaseType from spec')
    if name not in mapping:
        raise ValueError('Unknown base_type: %s' % name)
    return mapping[name]


def _spec_path_fallback():
    """Path to JSON next to this module (development and installed wheel)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, _DATA_SUBDIR, SPEC_FILENAME)


def _read_spec_bytes():
    p = _spec_path_fallback()
    if os.path.isfile(p):
        with open(p, 'rb') as f:
            return f.read()
    try:
        if hasattr(importlib_resources, 'files'):
            root = importlib_resources.files(_PACKAGE)
            path = root.joinpath(_DATA_SUBDIR, SPEC_FILENAME)
            return path.read_bytes()
    except (FileNotFoundError, OSError, TypeError, ValueError, AttributeError):
        pass
    raise FileNotFoundError(
        'Cannot find %s (looked beside fitwrite_spec and package %s)' % (SPEC_FILENAME, _PACKAGE)
    )


def load_fit_spec_raw():
    """Return spec as dict (no BaseType resolution)."""
    data = _read_spec_bytes()
    return json.loads(data.decode('utf-8'))


def _validate_spec(raw):
    version = raw.get('version', 0)
    if version != 1:
        warnings.warn(
            'fit_export_spec.json version %s may be unsupported (expected 1)' % version,
            UserWarning,
            stacklevel=2,
        )
    seen = set()
    for row in raw['developer_fields']:
        fid = row['field_id']
        if fid in seen:
            raise ValueError('Duplicate field_id in FIT spec: %s' % fid)
        seen.add(fid)
        sc = int(row.get('scale', 1))
        if sc < 0 or sc > 255:
            raise ValueError(
                'FIT field description scale must be 0-255 (field_id=%s scale=%s)' % (fid, sc)
            )
    return raw


def _materialize_dev_field_tuples(raw):
    """Resolve BaseType; build tuple lists consumed by fitwrite (requires fit-tool)."""
    rowing = []
    oarlock = []
    peak = []
    instroke_axis = []
    dual_sides = {}

    for row in raw['developer_fields']:
        group = row['group']
        bt = _base_type_from_string(row['base_type'])
        fid = row['field_id']
        name = row['fit_name']
        size = int(row['size'])
        scale = int(row['scale'])
        units = row.get('units', '') or ''
        cols = row.get('df_columns') or []

        if group == 'rowing':
            if len(cols) < 1:
                raise ValueError('rowing group requires at least one df_column (field_id=%s)' % fid)
            rowing.append((fid, list(cols), name, bt, size, scale, units))
        elif group == 'oarlock':
            oarlock.append((fid, list(cols), name, bt, size, scale, units))
        elif group == 'oarlock_dual':
            pair_key = row.get('dual_pair')
            side = row.get('side')
            if not pair_key or side not in ('port', 'starboard'):
                raise ValueError(
                    'oarlock_dual requires dual_pair and side=port|starboard (field_id=%s)' % fid
                )
            tup = (fid, list(cols), name, bt, size, scale, units)
            if pair_key not in dual_sides:
                dual_sides[pair_key] = {}
            if side in dual_sides[pair_key]:
                raise ValueError('duplicate oarlock_dual side for pair %s' % pair_key)
            dual_sides[pair_key][side] = tup
        elif group == 'peak_position':
            transformer = row.get('transformer')
            clip_max = row.get('clip_uint16_max')
            peak.append(
                (fid, list(cols), name, bt, size, scale, units, transformer, clip_max)
            )
        elif group == 'instroke_axis':
            instroke_axis.append((fid, name, bt, size, scale, units))
        else:
            raise ValueError('Unknown developer_fields group: %s' % group)

    oarlock_dual_pairs = []
    for pair_key, sides in sorted(dual_sides.items()):
        if 'port' not in sides or 'starboard' not in sides:
            raise ValueError(
                'oarlock_dual pair %r must have both port and starboard fields' % pair_key
            )
        oarlock_dual_pairs.append((pair_key, sides['port'], sides['starboard']))

    instroke_axis.sort(key=lambda x: x[0])

    return {
        'ROWING_DEV_FIELDS': rowing,
        'OARLOCK_DEV_FIELDS': oarlock,
        'OARLOCK_DUAL_PAIRS': oarlock_dual_pairs,
        'PEAK_POSITION_DEV_FIELDS': peak,
        'INSTROKE_AXIS_DEV_FIELDS': instroke_axis,
    }


def load_fit_spec():
    """
    Load and validate FIT export spec; cache globally.
    Returns dict with keys: raw, ROWING_DEV_FIELDS, OARLOCK_DEV_FIELDS, OARLOCK_DUAL_PAIRS,
    PEAK_POSITION_DEV_FIELDS, INSTROKE_AXIS_DEV_FIELDS, instroke_column_map,
    abscissa_enum, instroke_axis_field_ids, always_emit_field_ids, instroke_dynamic.
    """
    global _SPEC_CACHE
    if _SPEC_CACHE is not None:
        return _SPEC_CACHE

    raw = _validate_spec(load_fit_spec_raw())
    if _get_base_type_map() is None:
        raise ImportError('fit-tool is required to resolve FIT BaseType from spec')

    mats = _materialize_dev_field_tuples(raw)
    _SPEC_CACHE = {
        'raw': raw,
        'version': raw.get('version', 1),
        'instroke_column_map': dict(raw['instroke_column_map']),
        'abscissa_enum': dict(raw['abscissa_enum']),
        'instroke_axis_field_ids': tuple(raw['instroke_axis_field_ids']),
        'always_emit_field_ids': frozenset(raw['always_emit_field_ids']),
        'instroke_dynamic': dict(raw['instroke_dynamic']),
        **mats,
    }
    return _SPEC_CACHE


def get_abscissa_constants(spec=None):
    """Map spec abscissa_enum keys to legacy INSTROKE_ABSCISSA_* names."""
    if spec is None:
        spec = load_fit_spec()
    ae = spec['abscissa_enum']
    return {
        'INSTROKE_ABSCISSA_UNKNOWN': ae['UNKNOWN'],
        'INSTROKE_ABSCISSA_TIME_UNIFORM_MS': ae['TIME_UNIFORM_MS'],
        'INSTROKE_ABSCISSA_HANDLE_DISTANCE_UNIFORM_M': ae['HANDLE_DISTANCE_UNIFORM_M'],
        'INSTROKE_ABSCISSA_OAR_ANGLE_UNIFORM_DEG': ae['OAR_ANGLE_UNIFORM_DEG'],
        'INSTROKE_ABSCISSA_NORMALIZED_DRIVE_0_1': ae['NORMALIZED_DRIVE_0_1'],
    }
