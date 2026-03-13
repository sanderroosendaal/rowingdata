#!/bin/sh
# Run: pip install -r requirements-publish.txt  (twine, sphinx) before first publish
set -e
pip install -e .
rm -rf dist/
python setup.py sdist
twine upload --skip-existing dist/*
python setup.py build_sphinx -E 2>/dev/null || true
