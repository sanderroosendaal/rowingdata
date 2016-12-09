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

def get_file_type(f):
    fop = open(f,'r')
    extension = f[-3:].lower()
    if extension == 'csv':
	# get first and 7th line of file
	firstline = fop.readline()
	
	for i in range(3):
	    fourthline = fop.readline()

	for i in range(3):
	    seventhline = fop.readline()

	fop.close()

	if 'SpeedCoach GPS Pro' in fourthline:
	    return 'speedcoach2'

	if 'Practice Elapsed Time (s)' in firstline:
	    return 'mystery'

        if 'Club' in firstline:
            return 'boatcoach'
        
	if 'Hair' in seventhline:
	    return 'rp'

	if 'Total elapsed time (s)' in firstline:
	    return 'ergstick'

	if 'Stroke Number' in firstline:
	    return 'ergdata'

	if ' DriveTime (ms)' in firstline:
	    return 'csv'

	if 'HR' in firstline and 'Interval' in firstline and 'Avg HR' not in firstline:
	    return 'speedcoach'

	if 'stroke.REVISION' in firstline:
	    return 'painsleddesktop'

    if extension == 'tcx':
	try:
	    tree = objectify.parse(f)
	    rt = tree.getroot()
	except:
	    return 'unknown'

	if 'HeartRateBpm' in etree.tostring(rt):
	    return 'tcx'
	else:
	    return 'tcxnohr'

    if extension =='fit':
	try:
	    FitFile(f,check_crc=False).parse()
	except:
	    return 'unknown'

	return 'fit'
	    
    return 'unknown'
	

def get_file_line(linenr,f):
    fop = open(f,'r')
    for i in range(linenr):
	line = fop.readline()

    fop.close()
    return line


def skip_variable_footer(f):
    counter = 0
    counter2 = 0

    fop = open(f,'r')
    for line in fop:
	if line.startswith('Type') and counter>15:
	    counter2 = counter
	    counter += 1
	else:
	    counter += 1

    fop.close()
    return counter-counter2+1

def get_rowpro_footer(f,converters={}):
    counter = 0
    counter2 = 0

    fop = open(f,'r')
    for line in fop:
	if line.startswith('Type') and counter>15:
	    counter2 = counter
	    counter += 1
	else:
	    counter += 1

    fop.close()
    
    return pd.read_csv(f,skiprows=counter2,
		       converters=converters,
		       engine='python',
		       sep=None)
    

def skip_variable_header(f):
    counter = 0

    fop = open(f,'r')
    for line in fop:
	if line.startswith('Session Detail Data') or line.startswith('Per-Stroke Data'):
	    counter2 = counter
	else:
	    counter +=1

    fop.close()
    return counter2+2

def totimestamp(dt, epoch=datetime.datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6

def make_cumvalues_array(xvalues):
    """ Takes a Pandas dataframe with one column as input value.
    Tries to create a cumulative series.
    
    """
    
    newvalues = 0.0*xvalues
    dx = np.diff(xvalues)
    dxpos = dx
    nrsteps = len(dxpos[dxpos<0])
    lapidx = np.append(0,np.cumsum((-dx+abs(dx))/(-2*dx)))
    if (nrsteps>0):
	indexes = np.where(dxpos<0)
	for index in indexes:
	    dxpos[index] = xvalues[index+1]
	newvalues = np.append(0,np.cumsum(dxpos))+xvalues[0]
    else:
	newvalues = xvalues

    return [newvalues,abs(lapidx)]

def make_cumvalues(xvalues):
    """ Takes a Pandas dataframe with one column as input value.
    Tries to create a cumulative series.
    
    """
    
    newvalues = 0.0*xvalues
    dx = xvalues.diff()
    dxpos = dx
    mask = -xvalues.diff()>0.9*xvalues
    nrsteps = len(dx.loc[mask])
    lapidx = np.cumsum((-dx+abs(dx))/(-2*dx))
    lapidx = lapidx.fillna(value=0)
    if (nrsteps>0):
	dxpos[mask] = xvalues[mask]
	newvalues = np.cumsum(dxpos)+xvalues.ix[0,0]
	newvalues.ix[0,0] = xvalues.ix[0,0]
    else:
	newvalues = xvalues

    newvalues.fillna(method='ffill')

    return [newvalues,lapidx]

def timestrtosecs(string):
    dt = parser.parse(string,fuzzy=True)
    secs = 3600*dt.hour+60*dt.minute+dt.second

    return secs

def timestrtosecs2(timestring):
    try:
	h,m,s = timestring.split(':')
	sval = 3600*int(h)+60.*int(m)+float(s)
    except ValueError:
        try:
	    m,s = timestring.split(':')
	    sval = 60.*int(m)+float(s)
        except ValueError:
            sval = 0
        
    return sval


def getcol(df,column='TimeStamp (sec)'):
    if column:
        try:
            return df[column]
        except KeyError:
            pass

    l = len(df.index)
    return Series(np.zeros(l))
        

class CSVParser(object):
    """ Parser for reading CSV files created by Painsled

    """
    def __init__(self, *args, **kwargs):
        try:
            csvfile = args[0]
        except KeyError:
            csvfile = kwargs.pop('csvfile','test.csv')

        skiprows = kwargs.pop('skiprows',0)
        usecols = kwargs.pop('usecols',None)
        sep = kwargs.pop('sep',',')
        engine = kwargs.pop('engine','c')
            
        self.df = pd.read_csv(csvfile,skiprows=skiprows,usecols=usecols,
                              sep=sep,engine=engine)

        self.defaultcolumnnames = [
            'TimeStamp (sec)',
	    ' Horizontal (meters)',
	    ' Cadence (stokes/min)',
	    ' HRCur (bpm)',
	    ' Stroke500mPace (sec/500m)',
	    ' Power (watts)',
	    ' DriveLength (meters)',
	    ' StrokeDistance (meters)',
	    ' DriveTime (ms)',
	    ' DragFactor',
	    ' StrokeRecoveryTime (ms)',
	    ' AverageDriveForce (lbs)',
	    ' PeakDriveForce (lbs)',
	    ' lapIdx',
	    ' ElapsedTime (sec)',
            ' latitude',
            ' longitude',
	]

    def time_values(self,*args,**kwargs):
        timecolumn = kwargs.pop('timecolumn','TimeStamp (sec)')
        unixtimes = self.df[timecolumn]

        return unixtimes

    def write_csv(self,*args, **kwargs):
        gzip = kwargs.pop('gzip',False)
        writeFile = args[0]
        
        defaultmapping  = {c:c for c in self.defaultcolumnnames}
        columns = kwargs.pop('columns',defaultmapping)

	unixtimes = self.time_values(timecolumn=columns['TimeStamp (sec)'])
        self.df[columns['TimeStamp (sec)']] = unixtimes
        self.df[columns[' ElapsedTime (sec)']] = unixtimes-unixtimes.iloc[0]
        
        datadict = {name:getcol(self.df,columns[name]) for name in columns}

	nr_rows = len(self.df[columns[' Cadence (stokes/min)']])

	# Create data frame with all necessary data to write to csv
	data = DataFrame(datadict)

	data = data.sort_values(by='TimeStamp (sec)',ascending=True)
	
        if gzip:
	    return data.to_csv(writeFile+'.gz',index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writeFile,index_label='index')


        
class painsledDesktopParser(CSVParser):

    
    def __init__(self, *args, **kwargs):
        super(painsledDesktopParser, self).__init__(*args, **kwargs)
	# remove "00 waiting to row"
	self.df = self.df[self.df[' stroke.endWorkoutState'] != ' "00 waiting to row"']

        self.cols = [
            ' stroke.driveStartMs',
            ' stroke.startWorkoutMeter',
            ' stroke.strokesPerMin',
            ' stroke.hrBpm',
            ' stroke.paceSecPer1k',
            ' stroke.watts',
            ' stroke.driveMeters',
            ' stroke.strokeMeters',
            ' stroke.driveMs',
            ' stroke.dragFactor',
            ' stroke.slideMs',
            '',
            '',
            ' stroke.intervalNumber',
            ' stroke.driveStartMs',
            ' latitude',
            ' longitude',
        ]

        self.columns = dict(zip(self.defaultcolumnnames,self.cols))

        # calculations
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]/2.
        pace = np.clip(pace,0,1e4)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace
        timestamps = self.df[self.columns['TimeStamp (sec)']]
	# convert to unix style time stamp
	tts = timestamps.apply(lambda x:iso8601.parse_date(x[2:-1]))
        unixtimes = tts.apply(lambda x:time.mktime(x.timetuple()))
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes-unixtimes.iloc[0]


    def write_csv(self,*args,**kwargs):
        kwargs['columns'] = self.columns
        return super(painsledDesktopParser,self).write_csv(*args,**kwargs)

class BoatCoachParser(CSVParser):

    def __init__(self, *args, **kwargs):
        kwargs['skiprows']=1
        kwargs['usecols']=range(25)

        try:
            csvfile = args[0]
        except KeyError:
            csvfile = kwargs['csvfile']
            
        super(BoatCoachParser, self).__init__(*args, **kwargs)

        self.cols = [
            'workTime',
            'workDistance',
            'strokeRate',
            'currentHeartRate',
            'stroke500MPace',
            'strokePower',
            'strokeLength',
            '',
            'strokeDriveTime',
            'dragFactor',
            '',
            'strokeAverageForce',
            'strokePeakForce',
            'intervalCount',
            'workTime',
            ' latitude',
            ' longitude',
        ]

        self.columns = dict(zip(self.defaultcolumnnames,self.cols))

        # calculations
        # get date from footer
        fop = open(csvfile,'r')
        line = fop.readline()
        dated =  re.split('Date:',line)[1][1:-1]
	row_date = parser.parse(dated,fuzzy=True)
        fop.close()
        row_date2 = time.mktime(row_date.timetuple())
        timecolumn = self.df[self.columns['TimeStamp (sec)']]
        timesecs = timecolumn.apply(lambda x:timestrtosecs(x))
        timesecs = make_cumvalues(timesecs)[0]
        unixtimes = row_date2+timesecs

        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'

        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes-unixtimes[0]
        
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']].apply(lambda x:timestrtosecs(x))
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

    def write_csv(self,*args,**kwargs):
        kwargs['columns'] = self.columns
        return super(BoatCoachParser,self).write_csv(*args,**kwargs)



class ErgDataParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(ErgDataParser, self).__init__(*args, **kwargs)

        self.row_date = kwargs.pop('row_date',datetime.datetime.utcnow())
        self.cols = [
            'Time (seconds)',
            'Distance (meters)',
            'Stroke Rate',
            'Heart Rate',
            'Pace (seconds per 500m',
            ' Power (watts)',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            ' lapIdx',
            'Time(sec)',
            ' latitude',
            ' longitude',
        ]

        try:
            pace = self.df[self.cols[4]]
        except KeyError:
            self.cols[4] = 'Pace (seconds per 500m)'
            
        self.columns = dict(zip(self.defaultcolumnnames,self.cols))
                
        
        # calculations
        # get date from footer
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = np.clip(pace,0,1e4)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        seconds = self.df[self.columns['TimeStamp (sec)']]
        dt = seconds.diff()
        nrsteps = len(dt[dt<0])
        res = make_cumvalues(seconds)
        seconds2 = res[0]+seconds[0]
        lapidx = res[1]
        unixtime = seconds2+totimestamp(self.row_date)

        velocity = 500./pace
        power = 2.8*velocity**3

        self.df[self.columns['TimeStamp (sec)']] = unixtime
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtime-unixtime[0]

        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns[' Power (watts)']] = power
        
    def write_csv(self,*args,**kwargs):
        kwargs['columns'] = self.columns
        return super(ErgDataParser,self).write_csv(*args,**kwargs)

class speedcoachParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(speedcoachParser, self).__init__(*args, **kwargs)

        self.row_date = kwargs.pop('row_date',datetime.datetime.utcnow())
        self.cols = [
            'Time(sec)',
            'Distance(m)',
            'Rate',
            'HR',
            'Split(sec)',
            ' Power (watts)',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            ' lapIdx',
            'Time(sec)',
            ' latitude',
            ' longitude',
        ]

        self.columns = dict(zip(self.defaultcolumnnames,self.cols))
                
        
        # calculations
        # get date from footer
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = np.clip(pace,0,1e4)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        seconds = self.df[self.columns['TimeStamp (sec)']]
        unixtime = seconds+totimestamp(self.row_date)


        self.df[self.columns['TimeStamp (sec)']] = unixtime
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes-unixtimes[0]



    def write_csv(self,*args,**kwargs):
        kwargs['columns'] = self.columns
        return super(speedcoachParser,self).write_csv(*args,**kwargs)

class ErgStickParser(CSVParser):

    
    def __init__(self, *args, **kwargs):
        super(ErgStickParser, self).__init__(*args, **kwargs)

        self.row_date = kwargs.pop('row_date',datetime.datetime.utcnow())
        self.cols = [
            'Total elapsed time (s)',
            'Total distance (m)',
            'Stroke rate (/min)',
            'Current heart rate (bpm)',
            'Current pace (/500m)',
            ' Power (watts)',
            'Drive length (m)',
            'Stroke distance (m)',
            'Drive time (s)',
            'Drag factor',
            'Stroke recovery time (s)',
            'Ave. drive force (lbs)',
            'Peak drive force (lbs)',
            ' lapIdx',
            'Total elapsed time (s)',
            ' latitude',
            ' longitude',
        ]

        self.columns = dict(zip(self.defaultcolumnnames,self.cols))

        # calculations
        self.df[self.columns[' DriveTime (ms)']] *= 1000.
        self.df[self.columns[' StrokeRecoveryTime (ms)']] *= 1000.
        
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = np.clip(pace,1,1e4)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        velocity = 500./pace
        power = 2.8*velocity**3

        self.df[' Power (watts)'] = power

        seconds = self.df[self.columns['TimeStamp (sec)']]
        res = make_cumvalues(seconds)
        seconds2 = res[0]+seconds[0]
        lapidx = res[1]
        unixtimes = seconds2+totimestamp(self.row_date)
        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes-unixtimes.iloc[0]


    def write_csv(self,*args,**kwargs):
        kwargs['columns'] = self.columns
        return super(ErgStickParser,self).write_csv(*args,**kwargs)

class MysteryParser(CSVParser):

    
    def __init__(self, *args, **kwargs):
        super(MysteryParser, self).__init__(*args, **kwargs)
        self.df = self.df.drop(self.df.index[[0]])
        self.row_date = kwargs.pop('row_date',datetime.datetime.utcnow())
        
        kwargs['engine'] = 'python'
        kwargs['sep'] = None
        
        self.row_date = kwargs.pop('row_date',datetime.datetime.utcnow())
        self.cols = [
            'Practice Elapsed Time (s)',
            'Distance (m)',
            'Stroke Rate (SPM)',
            'HR (bpm)',
            ' Stroke500mPace (sec/500m)',
	    ' Power (watts)',
	    ' DriveLength (meters)',
	    ' StrokeDistance (meters)',
	    ' DriveTime (ms)',
	    ' DragFactor',
	    ' StrokeRecoveryTime (ms)',
	    ' AverageDriveForce (lbs)',
	    ' PeakDriveForce (lbs)',
	    ' lapIdx',
	    ' ElapsedTime (sec)',
            'Lat',
            'Lon',
        ]

        self.columns = dict(zip(self.defaultcolumnnames,self.cols))

        # calculations
        velo = pd.to_numeric(self.df['Speed (m/s)'],errors='coerce')
        
        pace = 500./velo
	pace = pace.replace(np.nan,300)
        pace = pace.replace(np.inf,300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        power = 2.8*velo**3
        self.df[' Power (watts)'] = power

        seconds = self.df[self.columns['TimeStamp (sec)']]
        res = make_cumvalues_array(np.array(seconds))
        seconds3 = res[0]
        lapidx = res[1]


        spm = self.df[self.columns[' Cadence (stokes/min)']]
        strokelength = velo/(spm/60.)
        
        unixtimes = pd.Series(seconds3+totimestamp(self.row_date))
        
        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes-unixtimes.iloc[0]
        self.df[self.columns[' StrokeDistance (meters)']] = strokelength


    def write_csv(self,*args,**kwargs):
        kwargs['columns'] = self.columns
        return super(MysteryParser,self).write_csv(*args,**kwargs)
