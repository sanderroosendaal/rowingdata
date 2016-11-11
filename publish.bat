python setup.py develop install
twine upload dist/*
python setup.py build_sphinx -E
