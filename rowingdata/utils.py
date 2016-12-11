import math
import numpy as np
import re
import time
import matplotlib
import iso8601
import os
import pickle
import pandas as pd
import datetime

def totimestamp(dt, epoch=datetime.datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6


def format_pace(x,pos=None):
    if np.isinf(x) or np.isnan(x):
	x=0
	
    min=int(x/60)
    sec=(x-min*60.)

    str1 = "{min:0>2}:{sec:0>4.1f}".format(
	min = min,
	sec = sec
	)

    return str1

def format_time(x,pos=None):


    min = int(x/60.)
    sec = int(x-min*60)

    str1 = "{min:0>2}:{sec:0>4.1f}".format(
	min=min,
	sec=sec,
	)

    return str1
