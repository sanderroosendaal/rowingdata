#!/bin/sh
python setup.py develop install
python setup.py sdist
twine upload --skip-existing dist/*
python setup.py build_sphinx -E
