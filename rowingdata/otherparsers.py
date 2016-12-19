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

from utils import *

namespace = 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'

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

	self.df = DataFrame({
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

    def write_csv(self,writeFile="fit_o.csv",gzip=False):

        if gzip:
            return self.df.to_csv(writeFile+'.gz',index_label='index',
                               compression='gzip')
        else:
	    return self.df.to_csv(writeFile,index_label='index')

	
class TCXParserTester:
    def __init__(self, tcx_file):
        tree = objectify.parse(tcx_file)
        self.root = tree.getroot()
        self.activity = self.root.Activities.Activity

	# need to select only trackpoints with Cadence, Distance, Time & HR data 
	self.selectionstring = '//ns:Trackpoint[descendant::ns:HeartRateBpm]'
	self.selectionstring +='[descendant::ns:Cadence]'
	self.selectionstring +='[descendant::ns:DistanceMeters]'
	self.selectionstring +='[descendant::ns:Time]'


	hr_values = self.root.xpath(self.selectionstring
			       +'//ns:HeartRateBpm/ns:Value',
			       namespaces={'ns': namespace})
        


        distance_values =  self.root.xpath(self.selectionstring
			       +'/ns:DistanceMeters',
			       namespaces={'ns': namespace})

	spm_values = self.root.xpath(self.selectionstring
			       +'/ns:Cadence',
			       namespaces={'ns': namespace})

    def getarray(self,str1,str2=''):
	s = self.selectionstring 
	s = s+'//ns:'+str1
	if (str2 != ''):
	    s = s+'/ns:'+str2

	y = self.root.xpath(s,namespaces={'ns': namespace})

	return y
	


class TCXParser:
    """ Parser for reading TCX files, e.g. from CrewNerd

    Use: data = rowingdata.TCXParser("crewnerd_data.tcx")

         data.write_csv("crewnerd_data_out.csv")

	 """

    
    def __init__(self, tcx_file):
        tree = objectify.parse(tcx_file)
        self.root = tree.getroot()
        self.activity = self.root.Activities.Activity

	# need to select only trackpoints with Cadence, Distance, Time & HR data 
	self.selectionstring = '//ns:Trackpoint[descendant::ns:HeartRateBpm]'
	self.selectionstring +='[descendant::ns:Cadence]'
	self.selectionstring +='[descendant::ns:DistanceMeters]'
	self.selectionstring +='[descendant::ns:Time]'


	hr_values = self.root.xpath(self.selectionstring
			       +'//ns:HeartRateBpm/ns:Value',
			       namespaces={'ns': namespace})
        


        distance_values =  self.root.xpath(self.selectionstring
			       +'/ns:DistanceMeters',
			       namespaces={'ns': namespace})

	spm_values = self.root.xpath(self.selectionstring
			       +'/ns:Cadence',
			       namespaces={'ns': namespace})


	# time stamps (ISO)
	timestamps = self.root.xpath(self.selectionstring
				    +'/ns:Time',
				    namespaces={'ns': namespace})
	
	lat_values = self.root.xpath(self.selectionstring
					  +'/ns:Position/ns:LatitudeDegrees',
					  namespaces={'ns':namespace})

	long_values = self.root.xpath(self.selectionstring
					   +'/ns:Position/ns:LongitudeDegrees',
					   namespaces={'ns':namespace})

	# and here are the trackpoints for "no stroke" 
	self.selectionstring2 = '//ns:Trackpoint[descendant::ns:HeartRateBpm]'
	self.selectionstring2 +='[descendant::ns:DistanceMeters]'
	self.selectionstring2 +='[descendant::ns:Time]'

	hr_values2 = self.root.xpath(self.selectionstring2
			       +'//ns:HeartRateBpm/ns:Value',
			       namespaces={'ns': namespace})
        


        distance_values2 =  self.root.xpath(self.selectionstring2
			       +'/ns:DistanceMeters',
			       namespaces={'ns': namespace})

	spm_values2 = np.zeros(len(distance_values2)).tolist()


	# time stamps (ISO)
	timestamps2 = self.root.xpath(self.selectionstring2
				    +'/ns:Time',
				    namespaces={'ns': namespace})
	
	lat_values2 = self.root.xpath(self.selectionstring2
					  +'/ns:Position/ns:LatitudeDegrees',
					  namespaces={'ns':namespace})

	long_values2 = self.root.xpath(self.selectionstring2
					   +'/ns:Position/ns:LongitudeDegrees',
					   namespaces={'ns':namespace})

	# merge the two datasets


	timestamps = timestamps+timestamps2
	
	self.hr_values = hr_values+hr_values2
	self.distance_values = distance_values+distance_values2

	self.spm_values = spm_values+spm_values2

	self.long_values = long_values+long_values2
	self.lat_values = lat_values+lat_values2

	# sort the two datasets
	data = pd.DataFrame({
	    't':timestamps,
	    'hr':self.hr_values,
	    'd':self.distance_values,
	    'spm':self.spm_values,
	    'long':self.long_values,
	    'lat':self.lat_values
	    })

	data = data.drop_duplicates(subset='t')
	data = data.sort_values(by='t',ascending = 1)

	timestamps = data.ix[:,'t'].values
	self.hr_values = data.ix[:,'hr'].values
	self.distance_values = data.ix[:,'d'].values
	self.spm_values = data.ix[:,'spm'].values
	self.long_values = data.ix[:,'long'].values
	self.lat_values = data.ix[:,'lat'].values

	# convert to unix style time stamp
	unixtimes = np.zeros(len(timestamps))

	# Activity ID timestamp (start)
	iso_string = str(self.root.Activities.Activity.Id)
	startdatetimeobj = iso8601.parse_date(iso_string)
	
	# startdatetimeobj = parser.parse(str(self.root.Activities.Activity.Id),fuzzy=True)
	starttime = time.mktime(startdatetimeobj.timetuple())+startdatetimeobj.microsecond/1.e6

	self.activity_starttime = starttime

	# there may be a more elegant and faster way with arrays 
	for i in range(len(timestamps)):
	    s = str(timestamps[i])
	    tt = iso8601.parse_date(s)
	    unixtimes[i] = time.mktime(tt.timetuple())+tt.microsecond/1.e6

	self.time_values = unixtimes
	
	long = self.long_values
	lat = self.lat_values
	spm = self.spm_values

	nr_rows = len(lat)
	velo = np.zeros(nr_rows)
	dist2 = np.zeros(nr_rows)
	strokelength = np.zeros(nr_rows)
	
	for i in range(nr_rows-1):
	    res = geo_distance(lat[i],long[i],lat[i+1],long[i+1])
	    dl = 1000.*res[0]
	    dist2[i+1]=dist2[i]+dl
	    velo[i+1] = dl/(1.0*(unixtimes[i+1]-unixtimes[i]))
	    if (spm[i]<>0):
		strokelength[i] = dl*60/spm[i]
	    else:
		strokelength[i] = 0.


	self.strokelength = strokelength
	self.dist2 = dist2
	self.velo = velo



    def write_csv(self,writeFile='example.csv',window_size=5,gzip=False):
	""" Exports TCX data to the CSV format that
	I use in rowingdata
	"""

	# Time stamps 
	unixtimes = self.time_values


	# Distance Meters
	d = self.distance_values

	# Stroke Rate
	spm = self.spm_values
	
	# Heart Rate
	hr = self.hr_values

	long = self.long_values
	lat = self.lat_values

	nr_rows = len(spm)
	velo = np.zeros(nr_rows)
	dist2 = np.zeros(nr_rows)
	strokelength = np.zeros(nr_rows)

	velo = self.velo
	strokelength = self.strokelength
	dist2 = self.dist2

	velo2 = ewmovingaverage(velo,window_size)
	strokelength2 = ewmovingaverage(strokelength,window_size)
	
	pace = 500./velo2
	pace = np.clip(pace,0,1e4)


	# Create data frame with all necessary data to write to csv
	data = DataFrame({'TimeStamp (sec)':unixtimes,
			  ' Horizontal (meters)': dist2,
			  ' Cadence (stokes/min)':spm,
			  ' HRCur (bpm)':hr,
			  ' longitude':long,
			  ' latitude':lat,
			  ' Stroke500mPace (sec/500m)':pace,
			  ' Power (watts)':np.zeros(nr_rows),
			  ' DriveLength (meters)':np.zeros(nr_rows),
			  ' StrokeDistance (meters)':strokelength2,
			  ' DragFactor':np.zeros(nr_rows),
			  ' DriveTime (ms)':np.zeros(nr_rows),
			  ' StrokeRecoveryTime (ms)':np.zeros(nr_rows),
			  ' AverageDriveForce (lbs)':np.zeros(nr_rows),
			  ' PeakDriveForce (lbs)':np.zeros(nr_rows),
			  ' lapIdx':np.zeros(nr_rows),
			  ' ElapsedTime (sec)':unixtimes-self.activity_starttime
			  })

	
	self.data = data

        if gzip:
	    return data.to_csv(writeFile+'.gz',index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writeFile,index_label='index')

	

    def write_nogeo_csv(self,writeFile='example.csv',window_size=5,gzip=False):
	""" Exports TCX data without position data (indoor)
	to the CSV format that
	I use in rowingdata
	"""

	# Time stamps 
	unixtimes = self.time_values


	# Distance Meters
	d = self.distance_values

	# Stroke Rate
	spm = self.spm_values
	
	# Heart Rate
	hr = self.hr_values


	nr_rows = len(spm)
	velo = np.zeros(nr_rows)

	strokelength = np.zeros(nr_rows)

	for i in range(nr_rows-1):
	    dl = d[i+1]-d[i]
	    if (unixtimes[i+1]<>unixtimes[i]):
		velo[i+1] = dl/(unixtimes[i+1]-unixtimes[i])
	    else:
		velo[i+1]=0

	    if (spm[i]<>0):
		strokelength[i] = dl*60/spm[i]
	    else:
		strokelength[i] = 0.


	velo2 = ewmovingaverage(velo,window_size)
	strokelength2 = ewmovingaverage(strokelength,window_size)
	pace = 500./velo2



	# Create data frame with all necessary data to write to csv
	data = DataFrame({'TimeStamp (sec)':unixtimes,
			  ' Horizontal (meters)': d,
			  ' Cadence (stokes/min)':spm,
			  ' HRCur (bpm)':hr,
			  ' Stroke500mPace (sec/500m)':pace,
			  ' Power (watts)':np.zeros(nr_rows),
			  ' DriveLength (meters)':np.zeros(nr_rows),
			  ' StrokeDistance (meters)':strokelength2,
			  ' DragFactor':dragfactor,
			  ' DriveTime (ms)':np.zeros(nr_rows),
			  ' StrokeRecoveryTime (ms)':np.zeros(nr_rows),
			  ' AverageDriveForce (lbs)':np.zeros(nr_rows),
			  ' PeakDriveForce (lbs)':np.zeros(nr_rows),
			  ' lapIdx':np.zeros(nr_rows),
			  ' ElapsedTime (sec)':unixtimes-self.activity_starttime
			  })
	
        if gzip:
	    return data.to_csv(writeFile+'.gz',index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writeFile,index_label='index')



class TCXParserNoHR:
    """ Parser for reading TCX files, e.g. from CrewNerd

    Use: data = rowingdata.TCXParser("crewnerd_data.tcx")

         data.write_csv("crewnerd_data_out.csv")

	 """

    
    def __init__(self, tcx_file):
        tree = objectify.parse(tcx_file)
        self.root = tree.getroot()
        self.activity = self.root.Activities.Activity

	# need to select only trackpoints with Cadence, Distance, Time & HR data 
	self.selectionstring = '//ns:Trackpoint[descendant::ns:Cadence]'
	self.selectionstring +='[descendant::ns:DistanceMeters]'
	self.selectionstring +='[descendant::ns:Time]'


        distance_values =  self.root.xpath(self.selectionstring
			       +'/ns:DistanceMeters',
			       namespaces={'ns': namespace})

	spm_values = self.root.xpath(self.selectionstring
			       +'/ns:Cadence',
			       namespaces={'ns': namespace})


	# time stamps (ISO)
	timestamps = self.root.xpath(self.selectionstring
				    +'/ns:Time',
				    namespaces={'ns': namespace})
	
	lat_values = self.root.xpath(self.selectionstring
					  +'/ns:Position/ns:LatitudeDegrees',
					  namespaces={'ns':namespace})

	long_values = self.root.xpath(self.selectionstring
					   +'/ns:Position/ns:LongitudeDegrees',
					   namespaces={'ns':namespace})

	# and here are the trackpoints for "no stroke" 
	self.selectionstring2 = '//ns:Trackpoint[descendant::ns:DistanceMeters]'
	self.selectionstring2 +='[descendant::ns:Time]'


        distance_values2 =  self.root.xpath(self.selectionstring2
			       +'/ns:DistanceMeters',
			       namespaces={'ns': namespace})

	spm_values2 = np.zeros(len(distance_values2)).tolist()


	# time stamps (ISO)
	timestamps2 = self.root.xpath(self.selectionstring2
				    +'/ns:Time',
				    namespaces={'ns': namespace})
	
	lat_values2 = self.root.xpath(self.selectionstring2
					  +'/ns:Position/ns:LatitudeDegrees',
					  namespaces={'ns':namespace})

	long_values2 = self.root.xpath(self.selectionstring2
					   +'/ns:Position/ns:LongitudeDegrees',
					   namespaces={'ns':namespace})

	# merge the two datasets


	timestamps = timestamps+timestamps2
	
	self.distance_values = distance_values+distance_values2

	self.spm_values = spm_values+spm_values2

	self.long_values = long_values+long_values2
	self.lat_values = lat_values+lat_values2

	# sort the two datasets
	data = pd.DataFrame({
	    't':timestamps,
	    'd':self.distance_values,
	    'spm':self.spm_values,
	    'long':self.long_values,
	    'lat':self.lat_values
	    })

	data = data.drop_duplicates(subset='t')
	data = data.sort_values(by='t',ascending = 1)

	timestamps = data.ix[:,'t'].values
	self.distance_values = data.ix[:,'d'].values
	self.spm_values = data.ix[:,'spm'].values
	self.long_values = data.ix[:,'long'].values
	self.lat_values = data.ix[:,'lat'].values

	# convert to unix style time stamp
	unixtimes = np.zeros(len(timestamps))

	# Activity ID timestamp (start)
	iso_string = str(self.root.Activities.Activity.Id)
	startdatetimeobj = iso8601.parse_date(iso_string)

	starttime = time.mktime(startdatetimeobj.timetuple())+startdatetimeobj.microsecond/1.e6

	self.activity_starttime = starttime

	# there may be a more elegant and faster way with arrays 
	for i in range(len(timestamps)):
	    s = str(timestamps[i])
	    tt = iso8601.parse_date(s)
	    unixtimes[i] = time.mktime(tt.timetuple())+tt.microsecond/1.e6

	self.time_values = unixtimes
	
	long = self.long_values
	lat = self.lat_values
	spm = self.spm_values

	nr_rows = len(lat)
	velo = np.zeros(nr_rows)
	dist2 = np.zeros(nr_rows)
	strokelength = np.zeros(nr_rows)
	
	for i in range(nr_rows-1):
	    res = geo_distance(lat[i],long[i],lat[i+1],long[i+1])
	    dl = 1000.*res[0]
	    dist2[i+1]=dist2[i]+dl
	    velo[i+1] = dl/(1.0*(unixtimes[i+1]-unixtimes[i]))
	    if (spm[i]<>0):
		strokelength[i] = dl*60/spm[i]
	    else:
		strokelength[i] = 0.


	self.strokelength = strokelength
	self.dist2 = dist2
	self.velo = velo



    def write_csv(self,writeFile='example.csv',window_size=5,gzip=False):
	""" Exports TCX data to the CSV format that
	I use in rowingdata
	"""

	# Time stamps 
	unixtimes = self.time_values


	# Distance Meters
	d = self.distance_values

	# Stroke Rate
	spm = self.spm_values
	
	# Heart Rate

	long = self.long_values
	lat = self.lat_values

	nr_rows = len(spm)
	velo = np.zeros(nr_rows)
	dist2 = np.zeros(nr_rows)
	strokelength = np.zeros(nr_rows)

	velo = self.velo
	strokelength = self.strokelength
	dist2 = self.dist2

	velo2 = ewmovingaverage(velo,window_size)
	strokelength2 = ewmovingaverage(strokelength,window_size)
	
	pace = 500./velo2
	pace = np.clip(pace,0,1e4)



	# Create data frame with all necessary data to write to csv
	data = DataFrame({'TimeStamp (sec)':unixtimes,
			  ' Horizontal (meters)': dist2,
			  ' Cadence (stokes/min)':spm,
			  ' HRCur (bpm)':np.zeros(nr_rows),
			  ' longitude':long,
			  ' latitude':lat,
			  ' Stroke500mPace (sec/500m)':pace,
			  ' Power (watts)':np.zeros(nr_rows),
			  ' DriveLength (meters)':np.zeros(nr_rows),
			  ' StrokeDistance (meters)':strokelength2,
			  ' DragFactor':np.zeros(nr_rows),
			  ' DriveTime (ms)':np.zeros(nr_rows),
			  ' StrokeRecoveryTime (ms)':np.zeros(nr_rows),
			  ' AverageDriveForce (lbs)':np.zeros(nr_rows),
			  ' PeakDriveForce (lbs)':np.zeros(nr_rows),
			  ' lapIdx':np.zeros(nr_rows),
			  ' ElapsedTime (sec)':unixtimes-self.activity_starttime
			  })

	
	self.data = data

        if gzip:
	    return data.to_csv(writeFile+'.gz',index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writeFile,index_label='index')

	

    def write_nogeo_csv(self,writeFile='example.csv',window_size=5,gzip=False):
	""" Exports TCX data without position data (indoor)
	to the CSV format that
	I use in rowingdata
	"""

	# Time stamps 
	unixtimes = self.time_values


	# Distance Meters
	d = self.distance_values

	# Stroke Rate
	spm = self.spm_values
	
	# Heart Rate
	hr = self.hr_values


	nr_rows = len(spm)
	velo = np.zeros(nr_rows)

	strokelength = np.zeros(nr_rows)

	for i in range(nr_rows-1):
	    dl = d[i+1]-d[i]
	    if (unixtimes[i+1]<>unixtimes[i]):
		velo[i+1] = dl/(unixtimes[i+1]-unixtimes[i])
	    else:
		velo[i+1]=0

	    if (spm[i]<>0):
		strokelength[i] = dl*60/spm[i]
	    else:
		strokelength[i] = 0.


	velo2 = ewmovingaverage(velo,window_size)
	strokelength2 = ewmovingaverage(strokelength,window_size)
	pace = 500./velo2



	# Create data frame with all necessary data to write to csv
	data = DataFrame({'TimeStamp (sec)':unixtimes,
			  ' Horizontal (meters)': d,
			  ' Cadence (stokes/min)':spm,
			  ' HRCur (bpm)':hr,
			  ' Stroke500mPace (sec/500m)':pace,
			  ' Power (watts)':np.zeros(nr_rows),
			  ' DriveLength (meters)':np.zeros(nr_rows),
			  ' StrokeDistance (meters)':strokelength2,
			  ' DragFactor':np.zeros(nr_rows),
			  ' DriveTime (ms)':np.zeros(nr_rows),
			  ' StrokeRecoveryTime (ms)':np.zeros(nr_rows),
			  ' AverageDriveForce (lbs)':np.zeros(nr_rows),
			  ' PeakDriveForce (lbs)':np.zeros(nr_rows),
			  ' lapIdx':np.zeros(nr_rows),
			  ' ElapsedTime (sec)':unixtimes-self.activity_starttime
			  })
	
        if gzip:
	    return data.to_csv(writeFile+'.gz',index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writeFile,index_label='index')

