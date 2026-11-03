"""Validator for the JSON Schema subset used by the detector_api_v1 contract (no external dependency).

Supported: $ref (#/$defs/X), type, const, enum, pattern, minimum, maximum, required, properties,
additionalProperties=false, items, minItems, maxItems, oneOf, allOf, if/then/else.
An unsupported keyword is an error, so the published schema cannot silently promise more than is checked.
"""
import math
import re

KNOWN = {'$schema', '$defs', '$ref', 'title', 'description', 'type', 'const', 'enum', 'pattern', 'minimum', 'maximum',
         'required', 'properties', 'additionalProperties', 'items', 'minItems', 'maxItems', 'oneOf', 'allOf',
         'if', 'then', 'else'}
TYPES = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'null': type(None), 'integer': int,
         'number': (int, float)}


class SchemaError(AssertionError):
    pass


def _fail(path, message):
    raise SchemaError('%s: %s' % (path or '$', message))


def valid(value, schema, root):
    try:
        check(value, schema, root)
        return True
    except SchemaError:
        return False


def check(value, schema, root=None, path=''):
    root = schema if root is None else root
    unknown = set(schema) - KNOWN
    if unknown:
        _fail(path, 'unsupported schema keywords %s' % sorted(unknown))
    if '$ref' in schema:
        check(value, root['$defs'][schema['$ref'].rsplit('/', 1)[-1]], root, path)
    for sub in schema.get('allOf', []):
        check(value, sub, root, path)
    if 'oneOf' in schema:
        matches = sum(valid(value, option, root) for option in schema['oneOf'])
        if matches != 1:
            _fail(path, 'matches %d of oneOf options' % matches)
    if 'if' in schema:
        branch = schema.get('then') if valid(value, schema['if'], root) else schema.get('else')
        if branch is not None:
            check(value, branch, root, path)
    if 'const' in schema and (value != schema['const'] or type(value) is not type(schema['const'])):
        _fail(path, 'const')
    if 'enum' in schema and value not in schema['enum']:
        _fail(path, 'enum')
    kinds = schema.get('type')
    if kinds:
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if not any(isinstance(value, TYPES[k]) and not (k in ('integer', 'number') and isinstance(value, bool)) for k in kinds):
            _fail(path, 'type %s' % kinds)
    if 'pattern' in schema and (not isinstance(value, str) or not re.search(schema['pattern'], value)):
        _fail(path, 'pattern')
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(value, float) and not math.isfinite(value):
        _fail(path, 'non-finite number')  # NaN/inf are never valid values, with or without bounds
    if 'minimum' in schema and numeric and value < schema['minimum']:
        _fail(path, 'minimum')
    if 'maximum' in schema and numeric and value > schema['maximum']:
        _fail(path, 'maximum')
    if isinstance(value, dict):
        for key in schema.get('required', []):
            if key not in value:
                _fail(path, 'missing ' + key)
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False and set(value) - set(properties):
            _fail(path, 'additional properties %s' % sorted(set(value) - set(properties)))
        for key, sub in properties.items():
            if key in value:
                check(value[key], sub, root, path + '.' + key)
    if isinstance(value, list):
        if 'minItems' in schema and len(value) < schema['minItems']:
            _fail(path, 'minItems')
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            _fail(path, 'maxItems')
        if 'items' in schema:
            for i, item in enumerate(value):
                check(item, schema['items'], root, '%s[%d]' % (path, i))
