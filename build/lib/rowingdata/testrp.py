from __future__ import absolute_import
from __future__ import print_function
from . import rowingdata
import os

def testrp(f,string=''):
    res = rowingdata.RowProParser('C:/Downloads/'+f)
    res.write_csv('C:/Downloads/'+f+'o.csv')
    r = rowingdata.rowingdata('C:/Downloads/'+f+'o.csv')
    print(r.allstats())
    if string != '':
	r.updateinterval_string(string)
	print(r.allstats())

    os.remove('C:/Downloads/'+f+'o.csv')
