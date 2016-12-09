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

def totimestamp(dt, epoch=datetime.datetime(1970,1,1)):
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6

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
            
        self.df = pd.read_csv(csvfile,skiprows=skiprows,usecols=usecols)

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
	    ' ElapsedTime (sec)'
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
        self.df[columns[' ElapsedTime (sec)']] = unixtimes-unixtimes[0]
        
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
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes-unixtimes[0]


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
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'Time(sec)',
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
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes-unixtimes[0]

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
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'Time(sec)',
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

