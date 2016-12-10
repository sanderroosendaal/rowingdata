import math
import numpy as np
import re
import time
import matplotlib
import iso8601
import os
import pickle
import pandas as pd
from pandas import Series,DataFrame
from dateutil import parser
import datetime
from lxml import objectify,etree
from fitparse import FitFile

from csvparsers import (
    totimestamp,
    )

class fitsummarydata:
    def __init__(self,readFile):
	self.readFile = readFile
	self.fitfile = FitFile(readFile,check_crc=False)
	self.records = self.fitfile.messages
	self.summarytext = 'Work Details\n'


    def setsummary(self,separator="|"):
	lapcount = 0
	self.summarytext += "#-{sep}SDist{sep}-Split-{sep}-SPace-{sep}-SPM-{sep}AvgHR{sep}MaxHR{sep}DPS-\n".format(
	    sep = separator
	    )

	strokecount = 0
	recordcount = 0
	totalhr = 0
	maxhr = 0

	totaldistance = 0
	totaltime = 0
	grandhr = 0
	grandmaxhr = 0
	
	for record in self.records:
	    if record.name == 'record':
		hr = record.get_value('heart_rate')
		if hr is None:
		    hr = 0
		if hr>maxhr:
		    maxhr = hr

		if hr>grandmaxhr:
		    grandmaxhr = hr
		    
		totalhr += hr

		grandhr += hr
		
		strokecount += 1
		recordcount += 1
		
	    if record.name == 'lap':
		lapcount += 1

		inthr = int(totalhr/float(strokecount))

		inttime = record.get_value('total_elapsed_time')

		lapmin = int(inttime/60)
		lapsec = int(int(10*(inttime-lapmin*60.))/10.)
		laptimestring = str(lapmin)+":"+str(lapsec)
		
		intdist = int(record.get_value('total_distance'))
		intvelo = intdist/inttime
		intpace = 500./intvelo

		totaldistance += intdist
		totaltime += inttime

		intspm = 60.*strokecount/inttime
		intdps = intdist/float(strokecount)

		intmaxhr = maxhr

		pacemin=int(intpace/60)
		pacesec=int(10*(intpace-pacemin*60.))/10.
		pacestring = str(pacemin)+":"+str(pacesec)

		strokecount = 0
		totalhr = 0
		maxhr = 0


		s = "{nr:0>2}{sep}{intdist:0>5d}{sep}".format(
		    nr = lapcount,
		    sep = separator,
		    intdist = intdist
		    )

		s += " {lapmin:0>2}:{lapsec:0>2} {sep}".format(
		    lapmin = lapmin,
		    lapsec = lapsec,
		    sep = separator,
		    )

		s+= "{pacemin:0>2}:{pacesec:0>3.1f}".format(
		    pacemin = pacemin,
		    pacesec = pacesec,
		    )

		s += "{sep} {intspm:0>4.1f}{sep}".format(
		    intspm = intspm,
		    sep = separator
		    )

		s += " {inthr:0>3d} {sep}".format(
		    inthr = inthr,
		    sep = separator
		    )

		s += " {intmaxhr:0>3d} {sep}".format(
		    intmaxhr = intmaxhr,
		    sep = separator
		    )

		s += " {dps:0>3.1f}".format(
		    dps = intdps
		    )

		s+= "\n"
		self.summarytext += s

	# add total summary
	overallvelo = totaldistance/totaltime
	overallpace = 500./overallvelo

	min=int(overallpace/60)
	sec=int(10*(overallpace-min*60.))/10.
	pacestring = str(min)+":"+str(sec)

	totmin = int(totaltime/60)
	totsec = int(int(10*(totaltime-totmin*60.))/10.)
	timestring = str(totmin)+":"+str(totsec)

	avghr = grandhr/float(recordcount)
	avgspm = 60.*recordcount/totaltime

	avgdps = totaldistance/float(recordcount)
	
	s = "Workout Summary\n"
	s += "--{sep}{totaldistance:0>5}{sep}".format(
	    totaldistance = int(totaldistance),
	    sep = separator
	    )
#	s += " "+timestring

	s += " {totmin:0>2}:{totsec:0>2} {sep} ".format(
	    totmin = totmin,
	    totsec = totsec,
	    sep = separator,
	    )
	    
	s += pacestring+separator

	s += " {avgspm:0>4.1f}{sep}".format(
	    sep = separator,
	    avgspm = avgspm
	    )

	s += " {avghr:0>3} {sep} {grandmaxhr:0>3} {sep}".format(
	    avghr = int(avghr),
	    grandmaxhr = int(grandmaxhr),
	    sep = separator
	    )

	s += " {avgdps:0>3.1f}".format(
	    avgdps = avgdps
	    )

	self.summarytext+=s
    
class FITParser:

    def __init__(self, readFile):
	self.readFile = readFile
	self.fitfile = FitFile(readFile,check_crc=False)
	self.records = self.fitfile.messages

    def write_csv(self,writeFile="fit_o.csv",gzip=False):
	cadence = []
	hr = []
	lat = []
	lon = []
	velo = []
	timestamp = []
	distance = []
	lapidx = []
	lapcounter = 0

	for record in self.records:
#	    if record.mesg_type.name == 'record':
	    if record.name == 'record':
		# obtain the values
		s = record.get_value('speed')
		h = record.get_value('heart_rate')
		spm = record.get_value('cadence')
		d = record.get_value('distance')
		t = record.get_value('timestamp')
		latva = record.get_value('position_lat')
		if latva is not None:
		    latv = record.get_value('position_lat')*(180./2**31)
		else:
		    latv = 0
		lonva = record.get_value('position_long')
		if lonva is not None:
		    lonv = record.get_value('position_long')*(180./2**31)
		else:
		    lonv = 0

		# get the unit


		# add the values to the list
		if latva is not None and lonva is not None:
		    velo.append(s)
		    hr.append(h)
		    lapidx.append(lapcounter)
		    lat.append(latv)
		    lon.append(lonv)
		    timestamp.append(totimestamp(t))
		    cadence.append(spm)
		    distance.append(d)
#	    if record.mesg_type.name == 'lap':
	    if record.name == 'lap':
		lapcounter += 1

	lat = pd.Series(lat)
	lon = pd.Series(lon)

	velo = pd.Series(velo)

	pace = 500./velo

	nr_rows = len(lat)

	seconds3 = np.array(timestamp)-timestamp[0]

	data = DataFrame({
	    'TimeStamp (sec)':timestamp,
	    ' Horizontal (meters)': distance,
	    ' Cadence (stokes/min)':cadence,
	    ' HRCur (bpm)':hr,
	    ' longitude':lon,
	    ' latitude':lat,
	    ' Stroke500mPace (sec/500m)':pace,
	    ' Power (watts)':np.zeros(nr_rows),
	    ' DriveLength (meters)':np.zeros(nr_rows),
	    ' StrokeDistance (meters)':np.zeros(nr_rows),
	    ' DriveTime (ms)':np.zeros(nr_rows),
	    ' DragFactor':np.zeros(nr_rows),
	    ' StrokeRecoveryTime (ms)':np.zeros(nr_rows),
	    ' AverageDriveForce (lbs)':np.zeros(nr_rows),
	    ' PeakDriveForce (lbs)':np.zeros(nr_rows),
	    ' ElapsedTime (sec)':seconds3,
	    ' lapIdx':lapidx,
	    })

        if gzip:
            return data.to_csv(writeFile+'.gz',index_label='index',
                               compression='gzip')
        else:
	    return data.to_csv(writeFile,index_label='index')
