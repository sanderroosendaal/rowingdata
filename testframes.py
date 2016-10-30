from math import *
from numpy import *
from pylab import *
from scipy import *
from matplotlib import *
import pandas as pd
from pandas import Series, DataFrame
import os
from Tkinter import Tk

import time
from lxml import objectify

def test():
    x = arange(15)
    y = x*x
    int_dist = array([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4])


    df = pd.DataFrame({'xvals':x,
		       'yvals':y,
		       'yvals2':zeros(15),
		       'int_dist':int_dist,
		       'cum_dist':zeros(15)})

    df['yvals2'][df['yvals']>50] = df['yvals']
    df['yvals2'][df['yvals']>100] = 0

    df['cum_dist'] = cumsum(df['int_dist'].diff()[df['int_dist'].diff()>0])

    print df
