python setup.py develop install
python setup.py sdist
twine upload dist/*
python setup.py build_sphinx -E
