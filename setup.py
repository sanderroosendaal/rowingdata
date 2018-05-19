from distutils.core import setup

import re

from rowingdata import rowingdata

def readme():
    with open('README.rst') as f:
        return f.read()

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('rowingdata/rowingdata.py').read(),
    re.M
).group(1)


requires = [
    'numpy',
    'scipy',
    'matplotlib',
    'pandas',
    'arrow',
    'python-dateutil',
    'docopt',
    'tqdm',
    'rowingphysics>=0.2.3',
    'iso8601',
    'lxml',
    'xmltodict',
    'nose_parameterized',
    'timezonefinder',
]
    
setup(
    name='rowingdata',
    version=version,
    description="The rowingdata library",
    long_description=readme(),
    url='https://github.com/sanderroosendaal/rowingdata',
    author='Sander Roosendaal',
    author_email='roosendaalsander@gmail.com',
    license='MIT',
    packages=['rowingdata'],
#    packages=find_packages(),
    keywords='rowing ergometer concept2',
    install_requires=requires,
)
