@echo off
REM Run: pip install -r requirements-publish.txt  (twine, sphinx) before first publish
pip install -e .
python setup.py sdist
twine upload --skip-existing dist/*
python setup.py build_sphinx -E 2>nul || echo Docs build skipped
