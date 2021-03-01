from __future__ import absolute_import
#from setuptools import setup, find_packages
from distutils.core import setup
import setuptools

import re

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='rowingdata',
      version=re.search(
          '^__version__\s*=\s*"(.*)"',
          open('rowingdata/rowingdata.py').read(),
          re.M
      ).group(1),

      classifiers = [
          "Programming Language :: Python :: 3",
          ],

      #      python_requires='>=3.4',

      description='The rowingdata library to create colorful plots from CrewNerd, Painsled and other rowing data tools',

      long_description=readme(),

      url='',

      author='Sander Roosendaal',

      author_email='roosendaalsander@gmail.com',

      license='MIT',

      packages=['rowingdata'],
      #packages=find_packages(),

      keywords='rowing ergometer concept2',

      install_requires=[
	  'Cython',
	  'numpy',
	  'scipy',
	  'matplotlib',
	  'pandas',
	  'fitparse',
	  #          'fitparse',
	  'arrow>=1.0.2',
	  #	  'mechanize',
	  'python-dateutil',
	  'docopt',
	  'tqdm',
	  'rowingphysics>=0.2.3',
	  'iso8601',
	  'lxml',
	  'xmltodict',
	  'nose_parameterized',
	  'timezonefinder',
	  'pycairo',
	  'tk','requests'
      ],

      zip_safe=False,
      include_package_data=True,
      # relative to the rowingdata directory
      package_data={
          'testdata':[
              'crewnerddata.CSV',
              'crewnerddata.tcx',
              'example.csv',
              'painsled_desktop_example.csv',
              'RP_testdata.csv',
              'SpeedCoach2example.csv',
              'testdata.csv'
          ],
          'bin':[
              'testdata.csv',
              'crewnerddata.csv',
              'crewnerddata.tcx',
          ],
          'rigging':[
              '1x.txt',
              '2x.txt',
              '4x-.txt',
              '4-.txt',
              '4+.txt',
              '2-.txt',
              '8+.txt'
          ]
      },

#      entry_points={
#          "console_scripts": [
#              'rowingdata=rowingdata.rowingdata:main',
#              'painsledtoc2=rowingdata.painsledtoc2:main',
#              'painsledplot=rowingdata.painsledplot:main',
#              'crewnerdplot=rowingdata.crewnerdplot:main',
#              'tcxtoc2=rowingdata.tcxtoc2:main',
#              'painsled_desktop_toc2=rowingdata.painsled_desktop_toc2:main',
#              'painsledplottime=rowingdata.painsledplottime:main',
#              'painsled_desktop_plottime=rowingdata.painsled_desktop_plottime:main',
#              'crewnerdplottime=rowingdata.crewnerdplottime:main',
#              'roweredit=rowingdata.roweredit:main',
#              'copystats=rowingdata.copystats:main',
#              'tcxplot=rowingdata.tcxplot:main',
#              'tcxplottime=rowingdata.tcxplottime:main',
#              'tcxplot_nogeo=rowingdata.tcxplot_nogeo:main',
#              'tcxplottime_nogeo=rowingdata.tcxplottime_nogeo:main',
#              'speedcoachplot=rowingdata.speedcoachplot:main',
#              'speedcoachplottime=rowingdata.speedcoachplottime:main',
#              'speedcoachtoc2=rowingdata.speedcoachtoc2:main',
#              'rowproplot=rowingdata.rowproplot:main',
#              'ergdataplot=rowingdata.ergdataplot:main',
#              'ergdataplottime=rowingdata.ergdataplottime:main',
#              'ergdatatotcx=rowingdata.ergdatatotcx:main',
#              'ergstickplot=rowingdata.ergstickplot:main',
#              'ergstickplottime=rowingdata.ergstickplottime:main',
#              'ergsticktotcx=rowingdata.ergsticktotcx:main',
#              'windcorrected=rowingdata.windcorrected:main',
#              'boatedit=rowingdata.boatedit:main',
#              'rowproplottime=rowingdata.rowproplottime:main'
#          ]
#      },


)
