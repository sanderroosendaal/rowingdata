# pylint: disable=C0103, C0303
import os
import csv
import gzip
import zipfile
import re
import datetime
import pytz
import arrow
import iso8601
import shutil
import numpy as np
import pandas as pd
from pandas.core.indexing import IndexingError

from StringIO import StringIO

from pandas import Series, DataFrame
from dateutil import parser

from timezonefinder import TimezoneFinder
from lxml import objectify
from fitparse import FitFile

from utils import (
    totimestamp, format_pace, format_time,
)

from tcxtools import strip_control_characters

# we're going to plot SI units - convert pound force to Newton
lbstoN = 4.44822

def clean_nan(x):
    for i in range(len(x) - 2):
        if np.isnan(x[i + 1]):
            if x[i + 2] > x[i]:
                x[i + 1] = 0.5 * (x[i] + x[i + 2])
            if x[i + 2] < x[i]:
                x[i + 1] = 0

    return x

def make_converter(convertlistbase,df):
    converters = {}
    for key in convertlistbase:
        try:
            try:
                values = df[key].apply(
                    lambda x: float(x.replace('.', '').replace(',', '.'))
                )
                converters[key] = lambda x: float(x.replace('.', '').replace(',', '.'))
            except AttributeError:
                pass
        except KeyError:
            pass

    return converters


def flexistrptime(inttime):

    try:
        t = datetime.datetime.strptime(inttime, "%H:%M:%S.%f")
    except ValueError:
        try:
            t = datetime.datetime.strptime(inttime, "%M:%S")
        except ValueError:
            try:
                t = datetime.datetime.strptime(inttime, "%H:%M:%S")
            except ValueError:
                t = datetime.datetime.strptime(inttime, "%M:%S.%f")

    return t

def flexistrftime(t):
    h = t.hour
    m = t.minute
    s = t.second
    us = t.microsecond

    second = s + us / 1.e6
    m = m + 60 * h
    string = "{m:0>2}:{s:0>4.1f}".format(
        m=m,
        s=second
    )

    return string

def csvtests(fop):
    # get first and 7th line of file
    firstline = fop.readline()
    secondline = fop.readline()
    thirdline = fop.readline()
    fourthline = fop.readline()

    for i in range(3):
        seventhline = fop.readline()

    fop.close()

    if 'Quiske' in firstline:
        return 'quiske'

    if 'RowDate' in firstline:
        return 'rowprolog'

    if 'Workout Name' in firstline:
        return 'c2log'

    if 'Concept2 Utility' in firstline:
        return 'c2log'

    if 'Concept2' in firstline:
        return 'c2log'

    if 'Workout #' in firstline:
        return 'c2log'

    if 'Activity Type' in firstline and 'Date' in firstline:
        return 'c2log'

    if 'Avg Watts' in firstline:
        return 'c2log'

    if 'SpeedCoach GPS Pro' in fourthline:
        return 'speedcoach2'

    if 'SpeedCoach GPS' in fourthline:
        return 'speedcoach2'

    if 'SpeedCoach GPS2' in fourthline:
        return 'speedcoach2'

    if 'SpeedCoach GPS Pro' in thirdline:
        return 'speedcoach2'

    if 'Practice Elapsed Time (s)' in firstline:
        return 'mystery'

    if 'Mike' in firstline and 'process' in firstline:
        return 'bcmike'

    if 'Club' in firstline and 'workoutType' in secondline:
        return 'boatcoach'

    if 'stroke500MPace' in firstline:
        return 'boatcoach'

    if 'Club' in secondline and 'Piece Stroke Count' in thirdline:
        return 'boatcoachotw'

    if 'peak_force_pos' in firstline:
        return 'rowperfect3'

    if 'Hair' in seventhline:
        return 'rp'

    if 'Total elapsed time (s)' in firstline:
        return 'ergstick'

    if 'Stroke Number' in firstline:
        return 'ergdata'

    if 'Number' in firstline and 'Cal/Hr' in firstline:
        return 'ergdata'

    if ' DriveTime (ms)' in firstline:
        return 'csv'

    if 'ElapsedTime (sec)' in firstline:
        return 'csv'

    if 'HR' in firstline and 'Interval' in firstline and 'Avg HR' not in firstline:
        return 'speedcoach'

    if 'stroke.REVISION' in firstline:
        return 'painsleddesktop'

    if 'Date' in firstline and 'Latitude' in firstline and 'Heart rate' in firstline:
        return 'kinomap'

    if 'Cover' in firstline:
        return 'coxmate'

    return 'unknown'

def get_file_type(f):
    extension = f[-3:].lower()
    if extension == 'xls':
        return 'xls'
    if extension == 'kml':
        return 'kml'
    if extension == '.gz':
        extension = f[-6:-3].lower()
        with gzip.open(f, 'r') as fop:
            try:
                if extension == 'csv':
                    return csvtests(fop)
                elif extension == 'tcx':
                    try:
                        input = fop.read()
                        input = strip_control_characters(input)
                        tree = objectify.parse(StringIO(input))
                        rt = tree.getroot()
                        return 'tcx'
                    except:
                        return 'unknown'
                elif extension == 'fit':
                    newfile = 'temp.fit'
                    with open(newfile,'wb') as f_out:
                        shutil.copyfileobj(fop, f_out)
  
                    try:
                        FitFile(newfile, check_crc=False).parse()
                        return 'fit'
                    except:
                        return 'unknown'

                    return 'fit'
                    
            except IOError:
                return 'notgzip'
    if extension == 'csv':
        if get_file_linecount(f) <= 2:
            return 'nostrokes'

        with open(f, 'r') as fop:
            return csvtests(fop)

    if extension == 'tcx':
        with open(f,'r') as fop:
            try:
                input = fop.read()
                input = strip_control_characters(input)
                tree = objectify.parse(StringIO(input))
                rt = tree.getroot()
                return 'tcx'
            except:
                return 'unknown'
            


        # if 'HeartRateBpm' in etree.tostring(rt):
        #    return 'tcx'
        # else:
        #    return 'tcxnohr'

    if extension == 'fit':
        try:
            FitFile(f, check_crc=False).parse()
        except:
            return 'unknown'

        return 'fit'

    if extension == 'zip':
        try:
            z = zipfile.ZipFile(f)
            f2 = z.extract(z.namelist()[0])
            tp = get_file_type(f2)
            os.remove(f2)
            return 'zip', f2, tp
        except:
            return 'unknown'

    return 'unknown'

def get_file_linecount(f):
    extension = f[-3:].lower()
    if extension == '.gz':
        with gzip.open(f,'rb') as fop:
            count = sum(1 for line in fop if line.rstrip('\n'))            
    else:
        with open(f, 'r') as fop:
            count = sum(1 for line in fop if line.rstrip('\n'))

    return count

def get_file_line(linenr, f):
    line = ''
    extension = f[-3:].lower()
    if extension == '.gz':
        with gzip.open(f, 'r') as fop:
            for i in range(linenr):
                line = fop.readline()
    else:
        with open(f, 'r') as fop:
            for i in range(linenr):
                line = fop.readline()

    return line


def get_separator(linenr, f):
    line = ''
    extension = f[-3:].lower()
    if extension == '.gz':    
        with gzip.open(f, 'r') as fop:
            for i in range(linenr):
                line = fop.readline()

            sep = ','
            sniffer = csv.Sniffer()
            sep = sniffer.sniff(line).delimiter
    else:
        with open(f, 'r') as fop:
            for i in range(linenr):
                line = fop.readline()

            sep = ','
            sniffer = csv.Sniffer()
            sep = sniffer.sniff(line).delimiter

    return sep

def getoarlength(line):
    l = float(line.split(',')[-1])
    return l

def getinboard(line):
    inboard = float(line.split(',')[-1])
    return inboard

def get_empower_rigging(f):
    oarlength = 289.
    inboard = 88.
    line = '1'
    with open(f, 'r') as fop:
        for line in fop:
            if 'Oar Length' in line:
                oarlength = getoarlength(line)
            if 'Inboard' in line:
                inboard = getinboard(line)
        

    return oarlength / 100., inboard / 100.

def skip_variable_footer(f):
    counter = 0
    counter2 = 0

    extension = f[-3:].lower()
    if extension == '.gz':
        fop = gzip.open(f,'rb')
    else:
        fop = open(f, 'r')
        
    for line in fop:
        if line.startswith('Type') and counter > 15:
            counter2 = counter
            counter += 1
        else:
            counter += 1

    fop.close()

    return counter - counter2 + 1

def get_rowpro_footer(f, converters={}):
    counter = 0
    counter2 = 0

    
    extension = f[-3:].lower()
    if extension == '.gz':
        fop = gzip.open(f,'rb')
    else:
        fop = open(f, 'r')

    for line in fop:
        if line.startswith('Type') and counter > 15:
            counter2 = counter
            counter += 1
        else:
            counter += 1

    fop.close()

    return pd.read_csv(f, skiprows=counter2,
                       converters=converters,
                       engine='python',
                       sep=None, index_col=False)


def skip_variable_header(f):
    counter = 0
    counter2 = 0
    summaryc = -2
    extension = f[-3:].lower()
    if extension == '.gz':
        fop = gzip.open(f,'rb')
    else:
        fop = open(f, 'r')


    for line in fop:
        if line.startswith('Interval Summaries'):
            summaryc = counter
        if line.startswith('Session Detail Data') or line.startswith('Per-Stroke Data'):
            counter2 = counter
        else:
            counter += 1

    fop.close()
            
    # test for blank line
    l = get_file_line(counter2 + 2, f)
    if 'Interval' in l:
        counter2 = counter2 - 1
        summaryc = summaryc - 1
        blanklines = 0
    else:
        blanklines = 1

    return counter2 + 2, summaryc + 2, blanklines

def make_cumvalues_array(xvalues):
    """ Takes a Pandas dataframe with one column as input value.
    Tries to create a cumulative series.

    """

    newvalues = 0.0 * xvalues
    dx = np.diff(xvalues)
    dxpos = dx
    nrsteps = len(dxpos[dxpos < 0])
    lapidx = np.append(0, np.cumsum((-dx + abs(dx)) / (-2 * dx)))
    if nrsteps > 0:
        indexes = np.where(dxpos < 0)
        for index in indexes:
            dxpos[index] = xvalues[index + 1]
        newvalues = np.append(0, np.cumsum(dxpos)) + xvalues[0]
    else:
        newvalues = xvalues

    return [newvalues, abs(lapidx)]

def make_cumvalues(xvalues):
    """ Takes a Pandas dataframe with one column as input value.
    Tries to create a cumulative series.

    """

    newvalues = 0.0 * xvalues
    dx = xvalues.diff()
    dxpos = dx
    mask = -xvalues.diff() > 0.9 * xvalues
    nrsteps = len(dx.loc[mask])
    lapidx = np.cumsum((-dx + abs(dx)) / (-2 * dx))
    lapidx = lapidx.fillna(value=0)
    test = len(lapidx.loc[lapidx.diff() < 0])
    if test != 0:
        lapidx = np.cumsum((-dx + abs(dx)) / (-2 * dx))
        lapidx = lapidx.fillna(method='ffill')
        lapidx.loc[0] = 0
    if nrsteps > 0:
        dxpos[mask] = xvalues[mask]
        try:
            newvalues = np.cumsum(dxpos) + xvalues.ix[0, 0]
            newvalues.ix[0, 0] = xvalues.ix[0, 0]
        except IndexingError:
            try:
                newvalues = np.cumsum(dxpos) + xvalues.iloc[0, 0]
                newvalues.iloc[0, 0] = xvalues.iloc[0, 0]
            except:
                newvalues = np.cumsum(dxpos)

    else:
        newvalues = xvalues

    newvalues = newvalues.replace([-np.inf, np.inf], np.nan)

    newvalues.fillna(method='ffill', inplace=True)
    lapidx.fillna(method='bfill', inplace=True)

    return [newvalues, lapidx]

def timestrtosecs(string):
    dt = parser.parse(string, fuzzy=True)
    secs = 3600 * dt.hour + 60 * dt.minute + dt.second

    return secs

def speedtopace(v, unit='ms'):
    if unit == 'kmh':
        v = v * 1000 / 3600.
    if v > 0:
        p = 500. / v
    else:
        p = np.nan

    return p

def timestrtosecs2(timestring, unknown=0):
    try:
        h, m, s = timestring.split(':')
        sval = 3600 * int(h) + 60. * int(m) + float(s)
    except ValueError:
        try:
            m, s = timestring.split(':')
            sval = 60. * int(m) + float(s)
        except ValueError:
            sval = unknown

    return sval


def getcol(df, column='TimeStamp (sec)'):
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
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs.pop('csvfile', 'test.csv')

        skiprows = kwargs.pop('skiprows', 0)
        usecols = kwargs.pop('usecols', None)
        sep = kwargs.pop('sep', ',')
        engine = kwargs.pop('engine', 'c')
        skipfooter = kwargs.pop('skipfooter', None)
        converters = kwargs.pop('converters', None)

        self.csvfile = csvfile


        if engine == 'python':
            self.df = pd.read_csv(
                csvfile, skiprows=skiprows, usecols=usecols,
                sep=sep, engine=engine, skipfooter=skipfooter,
                converters=converters, index_col=False,
                compression='infer',
            )
        else:
            self.df = pd.read_csv(
                csvfile, skiprows=skiprows, usecols=usecols,
                sep=sep, engine=engine, skipfooter=skipfooter,
                converters=converters, index_col=False,
                compression='infer',
                error_bad_lines = False
            )

            

        self.df = self.df.fillna(method='ffill')

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

        self.columns = {c: c for c in self.defaultcolumnnames}

    def to_standard(self):
        inverted = {value: key for key, value in self.columns.iteritems()}
        self.df.rename(columns=inverted, inplace=True)
        self.columns = {c: c for c in self.defaultcolumnnames}

    def time_values(self, *args, **kwargs):
        timecolumn = kwargs.pop('timecolumn', 'TimeStamp (sec)')
        unixtimes = self.df[timecolumn]

        return unixtimes

    def write_csv(self, *args, **kwargs):
        isgzip = kwargs.pop('gzip', False)
        writeFile = args[0]

        # defaultmapping ={c:c for c in self.defaultcolumnnames}
        self.columns = kwargs.pop('columns', self.columns)

        unixtimes = self.time_values(
            timecolumn=self.columns['TimeStamp (sec)'])
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.df[
            self.columns[' ElapsedTime (sec)']
        ] = unixtimes - unixtimes.iloc[0]
        # Default calculations
        pace = self.df[
            self.columns[' Stroke500mPace (sec/500m)']].replace(0, 300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        datadict = {name: getcol(self.df, self.columns[name])
                    for name in self.columns}

        # Create data frame with all necessary data to write to csv
        data = DataFrame(datadict)

        data = data.sort_values(by='TimeStamp (sec)', ascending=True)
        data = data.fillna(method='ffill')

        # drop all-zero columns
        for c in data.columns:
            if (data[c] == 0).any() and data[c].mean() == 0:
                data = data.drop(c, axis=1)

        if isgzip:
            return data.to_csv(writeFile + '.gz', index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writeFile, index_label='index')

class QuiskeParser(CSVParser):
    def __init__(self, *args, **kwargs):
        kwargs['skiprows'] = 1
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        super(QuiskeParser, self).__init__(*args, **kwargs)

        self.cols = [
            'timestamp(s)',
            'distance(m)',
            'SPM (strokes per minute)',
            '',
            ' Stroke500mPace (sec/500m)',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'latitude',
            'longitude',
            'Catch',
            'Finish',
            ]

        self.defaultcolumnnames += [
            'catch',
            'finish',
            ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        velo = self.df['speed (m/s)']
        pace = 500./velo
        pace = pace.replace(np.nan, 300)
        pace = pace.replace(np.inf, 300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace
        unixtimes = self.df[self.columns['TimeStamp (sec)']]
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes.iloc[0]
        self.df[self.columns['catch']] = 0
        self.df[self.columns['finish']] = self.df['stroke angle (deg)']
        
        self.to_standard()
            
class BoatCoachOTWParser(CSVParser):

    def __init__(self, *args, **kwargs):
        kwargs['skiprows'] = 2
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        ll = get_file_line(1,csvfile)
        if 'workoutType' in ll:
            kwargs['skiprows'] = 0

        separator = get_separator(3, csvfile)
        kwargs['sep'] = separator

        super(BoatCoachOTWParser, self).__init__(*args, **kwargs)

        # crude EU format detector
        try:
            ll = self.df['Last 10 Stroke Speed(/500m)']*10.0
        except TypeError:
            convertlistbase = [
                'TOTAL Distance Since Start BoatCoach(m)',
                'Stroke Rate',
                'Heart Rate',
                'Latitude',
                'Longitude',
            ]
            converters = make_converter(convertlistbase,self.df)
            
            kwargs['converters'] = converters
            super(BoatCoachOTWParser, self).__init__(*args, **kwargs)

        self.cols = [
            'DateTime',
            'TOTAL Distance Since Start BoatCoach(m)',
            'Stroke Rate',
            'Heart Rate',
            'Last 10 Stroke Speed(/500m)',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'Piece Number',
            'Elapsed Time',
            'Latitude',
            'Longitude',
        ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        try:
            row_datetime = self.df[self.columns['TimeStamp (sec)']]
            row_date = parser.parse(row_datetime[0], fuzzy=True)
            row_datetime = row_datetime.apply(
                lambda x: parser.parse(x, fuzzy=True))
            unixtimes = row_datetime.apply(lambda x: arrow.get(
                x).timestamp + arrow.get(x).microsecond / 1.e6)
        except KeyError:
            row_date2 = arrow.get(row_date).timestamp
            timecolumn = self.df[self.columns[' ElapsedTime (sec)']]
            timesecs = timecolumn.apply(timestrtosecs)
            timesecs = make_cumvalues(timesecs)[0]
            unixtimes = row_date2 + timesecs

        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'

        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes[0]
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']].apply(
            timestrtosecs2
        )
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        self.to_standard()

class CoxMateParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(CoxMateParser, self).__init__(*args, **kwargs)
        # remove "00 waiting to row"

        self.cols = [
            'Time',
            'Distance',
            'Rating',
            'Heart Rate',
            '',
            '',
            '',
            'Cover',
            '',
            '',
            '',
            '',
            '',
            '',
            'Time',
            '',
            '',
        ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # calculations / speed
        dd = self.df[self.columns[' Horizontal (meters)']].diff()
        dt = self.df[self.columns[' ElapsedTime (sec)']].diff()
        velo = dd / dt
        pace = 500. / velo
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        # calculations / time stamp

        # convert to unix style time stamp
        now = datetime.datetime.utcnow()
        elapsed = self.df[self.columns[' ElapsedTime (sec)']]
        tts = now + elapsed.apply(lambda x: datetime.timedelta(seconds=x))
        #unixtimes=tts.apply(lambda x: time.mktime(x.utctimetuple()))
        unixtimes = tts.apply(lambda x: arrow.get(
            x).timestamp + arrow.get(x).microsecond / 1.e6)
        self.df[self.columns['TimeStamp (sec)']] = unixtimes

        self.to_standard()


class painsledDesktopParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(painsledDesktopParser, self).__init__(*args, **kwargs)
        # remove "00 waiting to row"
        self.df = self.df[self.df[' stroke.endWorkoutState']
                          != ' "00 waiting to row"']

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

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # calculations
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']] / 2.
        pace = np.clip(pace, 0, 1e4)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace
        timestamps = self.df[self.columns['TimeStamp (sec)']]
        # convert to unix style time stamp
        tts = timestamps.apply(lambda x: iso8601.parse_date(x[2:-1]))
        #unixtimes=tts.apply(lambda x: time.mktime(x.utctimetuple()))
        unixtimes = tts.apply(lambda x: arrow.get(
            x).timestamp + arrow.get(x).microsecond / 1.e6)
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[
            self.columns[' ElapsedTime (sec)']
        ] = unixtimes - unixtimes.iloc[0]
        self.to_standard()


class BoatCoachParser(CSVParser):

    def __init__(self, *args, **kwargs):
        kwargs['skiprows'] = 1
        kwargs['usecols'] = range(25)

        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        ll = get_file_line(1,csvfile)
        if 'workoutType' in ll:
            kwargs['skiprows'] = 0

        separator = get_separator(2, csvfile)
        kwargs['sep'] = separator

        super(BoatCoachParser, self).__init__(*args, **kwargs)

        # crude EU format detector
        try:
            p = self.df['stroke500MPace'] * 500.
        except TypeError:
            convertlistbase = ['workDistance',
                           'strokeRate',
                           'currentHeartRate',
                           'strokePower',
                           'strokeLength',
                           'strokeDriveTime',
                           'dragFactor',
                           'strokeAverageForce',
                           'strokePeakForce',
                           'intervalCount']
            converters = make_converter(convertlistbase,self.df)

            kwargs['converters'] = converters
            super(BoatCoachParser, self).__init__(*args, **kwargs)

            
        self.cols = [
            'DateTime',
            'workDistance',
            'strokeRate',
            'currentHeartRate',
            'stroke500MPace',
            'strokePower',
            'strokeLength',
            '',
            'strokeDriveTime',
            'dragFactor',
            ' StrokeRecoveryTime (ms)',
            'strokeAverageForce',
            'strokePeakForce',
            'intervalCount',
            'workTime',
            ' latitude',
            ' longitude',
        ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # get date from footer
        try:
            try:
                with open(csvfile, 'r') as fop:
                    line = fop.readline()
                    dated = re.split('Date:', line)[1][1:-1]
            except IndexError:
                with gzip.open(csvfile,'rb') as fop:
                    line = fop.readline()
                    dated = re.split('Date:', line)[1][1:-1]
            row_date = parser.parse(dated, fuzzy=True)
        except IOError:
            pass
                

        try:
            datetime = self.df[self.columns['TimeStamp (sec)']]
            row_date = parser.parse(datetime[0], fuzzy=True)
            row_datetime = datetime.apply(lambda x: parser.parse(x, fuzzy=True))
            unixtimes = row_datetime.apply(
                lambda x: arrow.get(x).timestamp + arrow.get(x).microsecond / 1.e6
            )
        except KeyError:
            # calculations
            # row_date2=time.mktime(row_date.utctimetuple())
            row_date2 = arrow.get(row_date).timestamp
            timecolumn = self.df[self.columns[' ElapsedTime (sec)']]
            timesecs = timecolumn.apply(timestrtosecs)
            timesecs = make_cumvalues(timesecs)[0]
            unixtimes = row_date2 + timesecs

            
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'

        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes[0]

        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']].apply(
            timestrtosecs)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        self.df[self.columns[' DriveTime (ms)']] = 1.0e3 * \
            self.df[self.columns[' DriveTime (ms)']]

        drivetime = self.df[self.columns[' DriveTime (ms)']]
        stroketime = 60. * 1000. / \
            (1.0 * self.df[self.columns[' Cadence (stokes/min)']])
        recoverytime = stroketime - drivetime
        recoverytime.replace(np.inf, np.nan)
        recoverytime.replace(-np.inf, np.nan)
        recoverytime = recoverytime.fillna(method='bfill')

        self.df[self.columns[' StrokeRecoveryTime (ms)']] = recoverytime

        # Reset Interval Count by StrokeCount
        res = make_cumvalues(self.df['strokeCount'])
        lapidx = res[1]
        strokecount = res[0]
        self.df['strokeCount'] = strokecount
        if lapidx.max() > 1:
            self.df[self.columns[' lapIdx']] = lapidx

        # Recalculate power
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = np.clip(pace, 0, 1e4)
        pace = pace.replace(0, 300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace
        velocity = 500. / (1.0 * pace)
        power = 2.8 * velocity**3
        dif = abs(power - self.df[self.columns[' Power (watts)']])
        power[dif < 5] = self.df[self.columns[' Power (watts)']][dif < 5]
        self.df[self.columns[' Power (watts)']] = power

        # Calculate Stroke Rate during rest
        mask = (self.df['intervalType'] == 'Rest')
        for strokenr in self.df.loc[mask, 'strokeCount'].unique():
            mask2 = self.df['strokeCount'] == strokenr
            strokes = self.df.loc[mask2, 'strokeCount']
            timestamps = self.df.loc[mask2, self.columns['TimeStamp (sec)']]
            strokeduration = len(strokes) * timestamps.diff().mean()
            spm = 60. / strokeduration
            self.df.loc[mask2, self.columns[' Cadence (stokes/min)']] = spm


        # get stroke power
        data = []
        try:
            
            with gzip.open(csvfile,'r') as f:
                for line in f:
                    s  = line.split(',')
                    data.append(','.join([str(x) for x in s[26:-1]]))
        except IOError:
            with open(csvfile,'r') as f:
                for line in f:
                    s  = line.split(',')
                    data.append(','.join([str(x) for x in s[26:-1]]))

        try:
            self.df['PowerCurve'] = data[2:]
        except ValueError:
            pass

            
        # dump empty lines at end
        endhorizontal = self.df.loc[self.df.index[-1],
                                    self.columns[' Horizontal (meters)']]

        if endhorizontal == 0:
            self.df.drop(self.df.index[-1], inplace=True)

        res = make_cumvalues(self.df[self.columns[' Horizontal (meters)']])
        self.df['cumdist'] = res[0]
        maxdist = self.df['cumdist'].max()
        mask = (self.df['cumdist'] == maxdist)
        while len(self.df[mask]) > 2:
            mask = (self.df['cumdist'] == maxdist)
            self.df.drop(self.df.index[-1], inplace=True)

        mask = (self.df['cumdist'] == maxdist)
        self.df.loc[
            mask,
            self.columns[' lapIdx']
        ] = self.df.loc[self.df.index[-3], self.columns[' lapIdx']]

                
        self.to_standard()

class KinoMapParser(CSVParser):

    def __init__(self, *args, **kwargs):
        kwargs['skiprows'] = 0
        #kwargs['usecols'] = range(25)

        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        super(KinoMapParser, self).__init__(*args, **kwargs)

        self.cols = [
            'Date',
            'Distance',
            'Cadence',
            'Heart rate',
            'Speed',
            'Power',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'Latitude',
            'Longitude',
        ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        row_datetime = self.df[self.columns['TimeStamp (sec)']]
        row_datetime = row_datetime.apply(
            lambda x: parser.parse(x, fuzzy=True))
        #unixtimes=datetime.apply(lambda x: time.mktime(x.utctimetuple()))
        unixtimes = row_datetime.apply(lambda x: arrow.get(
            x).timestamp + arrow.get(x).microsecond / 1.e6)
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'

        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes[0]

        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']].apply(
            lambda x: speedtopace(x, unit='kmh'))
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        res = make_cumvalues(self.df[self.columns[' Horizontal (meters)']])
        self.df['cumdist'] = res[0]
        maxdist = self.df['cumdist'].max()
        mask = (self.df['cumdist'] == maxdist)
        while len(self.df[mask]) > 2:
            mask = (self.df['cumdist'] == maxdist)
            self.df.drop(self.df.index[-1], inplace=True)

        mask = (self.df['cumdist'] == maxdist)

        self.to_standard()


class BoatCoachAdvancedParser(CSVParser):

    def __init__(self, *args, **kwargs):
        kwargs['skiprows'] = 1
        kwargs['usecols'] = range(25)

        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        separator = get_separator(2, csvfile)
        kwargs['sep'] = separator
            

        super(BoatCoachAdvancedParser, self).__init__(*args, **kwargs)
        # crude EU format detector
        try:
            p = self.df['stroke500MPace'] * 500.
        except TypeError:
            convertlistbase = [
                'workDistance',
                'strokeRate',
                'currentHeartRate',
                'strokePower',
                'strokeLength',
                'strokeDriveTime',
                'dragFactor',
                'strokeAverageForce',
                'strokePeakForce',
                'intervalCount',
            ]
            
            converters = make_converter(convertlistbase,self.df)
            
            kwargs['converters'] = converters
            super(BoatCoachParser, self).__init__(*args, **kwargs)

        self.cols = [
            'DateTime',
            'workDistance',
            'strokeRate',
            'currentHeartRate',
            'stroke500MPace',
            'strokePower',
            'strokeLength',
            '',
            'strokeDriveTime',
            'dragFactor',
            ' StrokeRecoveryTime (ms)',
            'strokeAverageForce',
            'strokePeakForce',
            'intervalCount',
            'workTime',
            ' latitude',
            ' longitude',
        ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # get date from footer
        try:
            with open(csvfile, 'r') as fop:
                line = fop.readline()
                dated = re.split('Date:', line)[1][1:-1]
        except IndexError:
            with gzip.open(csvfile,'rb') as fop:
                line = fop.readline()
                dated = re.split('Date:', line)[1][1:-1]
            
        row_date = parser.parse(dated, fuzzy=True)

        try:
            row_datetime = self.df[self.columns['TimeStamp (sec)']]
            row_date = parser.parse(datetime[0], fuzzy=True)
            rowdatetime = row_datetime.apply(lambda x: parser.parse(x, fuzzy=True))
            unixtimes = row_datetime.apply(lambda x: arrow.get(
                x).timestamp + arrow.get(x).microsecond / 1.e6)
        except KeyError:
            # calculations
            # row_date2=time.mktime(row_date.utctimetuple())
            row_date2 = arrow.get(row_date).timestamp
            timecolumn = self.df[self.columns[' ElapsedTime (sec)']]
            timesecs = timecolumn.apply(timestrtosecs)
            timesecs = make_cumvalues(timesecs)[0]
            unixtimes = row_date2 + timesecs

        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'

        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes[0]

        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']].apply(
            timestrtosecs)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        self.df[self.columns[' DriveTime (ms)']] = 1.0e3 * \
            self.df[self.columns[' DriveTime (ms)']]

        # Calculate Recovery Time
        drivetime = self.df[self.columns[' DriveTime (ms)']]
        stroketime = 60. * 1000. / \
            (1.0 * self.df[self.columns[' Cadence (stokes/min)']])
        recoverytime = stroketime - drivetime
        recoverytime.replace(np.inf, np.nan)
        recoverytime.replace(-np.inf, np.nan)
        recoverytime = recoverytime.fillna(method='bfill')
        self.df[self.columns[' StrokeRecoveryTime (ms)']] = recoverytime

        # Reset Interval Count by StrokeCount
        res = make_cumvalues(self.df['strokeCount'])
        lapidx = res[1]
        strokecount = res[0]
        self.df['strokeCount'] = strokecount
        if lapidx.max() > 1:
            self.df[self.columns[' lapIdx']] = lapidx

        lapmax = self.df[self.columns[' lapIdx']].max()

        # Recalculate power
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = np.clip(pace, 0, 1e4)
        pace = pace.replace(0, 300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace
        velocity = 500. / pace
        power = 2.8 * velocity**3
        self.df[self.columns[' Power (watts)']] = power

        # Calculate Stroke Rate during rest
        mask = (self.df['intervalType'] == 'Rest')
        for strokenr in self.df.loc[mask, 'strokeCount'].unique():
            mask2 = self.df['strokeCount'] == strokenr
            strokes = self.df.loc[mask2, 'strokeCount']
            timestamps = self.df.loc[mask2, self.columns['TimeStamp (sec)']]
            strokeduration = len(strokes) * timestamps.diff().mean()
            spm = 60. / strokeduration
            self.df.loc[mask2, self.columns[' Cadence (stokes/min)']] = spm

        # dump empty lines at end
        endhorizontal = self.df.loc[self.df.index[-1],
                                    self.columns[' Horizontal (meters)']]

        if endhorizontal == 0:
            self.df.drop(self.df.index[-1], inplace=True)

        res = make_cumvalues(self.df[self.columns[' Horizontal (meters)']])
        self.df['cumdist'] = res[0]
        maxdist = self.df['cumdist'].max()
        mask = (self.df['cumdist'] == maxdist)
        while len(self.df[mask]) > 2:
            mask = (self.df['cumdist'] == maxdist)
            self.df.drop(self.df.index[-1], inplace=True)

        mask = (self.df['cumdist'] == maxdist)
        self.df.loc[
            mask,
            self.columns[' lapIdx']
        ] = self.df.loc[self.df.index[-3], self.columns[' lapIdx']]

        self.to_standard()


class ErgDataParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(ErgDataParser, self).__init__(*args, **kwargs)

        self.row_date = kwargs.pop('row_date', datetime.datetime.utcnow())
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
            try:
                pace = self.df[self.cols[4]]
            except KeyError:
                self.cols[4] = 'Pace (seconds)'

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # calculations
        # get date from footer
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = np.clip(pace, 0, 1e4)
        pace = pace.replace(0, 300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        seconds = self.df[self.columns['TimeStamp (sec)']]
        firststrokeoffset = seconds.values[0]
        res = make_cumvalues(seconds)
        seconds2 = res[0] + seconds[0]
        lapidx = res[1]
        unixtime = seconds2 + totimestamp(self.row_date)

        velocity = 500. / pace
        power = 2.8 * velocity**3

        self.df[self.columns['TimeStamp (sec)']] = unixtime
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtime - unixtime[0]
        self.df[self.columns[' ElapsedTime (sec)']] += firststrokeoffset

        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns[' Power (watts)']] = power

        self.to_standard()

class speedcoachParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(speedcoachParser, self).__init__(*args, **kwargs)

        self.row_date = kwargs.pop('row_date', datetime.datetime.utcnow())
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

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # calculations
        # get date from footer
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = np.clip(pace, 0, 1e4)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        seconds = self.df[self.columns['TimeStamp (sec)']]
        unixtimes = seconds + totimestamp(self.row_date)

        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes[0]

        self.to_standard()

class ErgStickParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(ErgStickParser, self).__init__(*args, **kwargs)

        self.row_date = kwargs.pop('row_date', datetime.datetime.utcnow())
        self.cols = [
            'Total elapsed time (s)',
            'Total distance (m)',
            'Stroke rate (/min)',
            'Current heart rate (bpm)',
            'Current pace (/500m)',
            'Split average power (W)',
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

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # calculations
        self.df[self.columns[' DriveTime (ms)']] *= 1000.
        self.df[self.columns[' StrokeRecoveryTime (ms)']] *= 1000.

        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = np.clip(pace, 1, 1e4)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        #velocity = 500. / pace
        #power = 2.8 * velocity**3

        #self.df[' Power (watts)'] = power

        seconds = self.df[self.columns['TimeStamp (sec)']]
        res = make_cumvalues(seconds)
        seconds2 = res[0] + seconds[0]
        lapidx = res[1]
        unixtimes = seconds2 + totimestamp(self.row_date)
        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes.iloc[0]

        self.to_standard()

class RowPerfectParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(RowPerfectParser, self).__init__(*args, **kwargs)

        for c in self.df.columns:
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        self.df.sort_values(by=['workout_interval_id', 'stroke_number'],
                            ascending=[True, True], inplace=True)

        
        self.row_date = kwargs.pop('row_date', datetime.datetime.utcnow())
        self.cols = [
            'time',
            'distance',
            'stroke_rate',
            'pulse',
            '',
            'power',
            'stroke_length',
            'distance_per_stroke',
            'drive_time',
            'k',
            'recover_time',
            '',
            'peak_force',
            'workout_interval_id',
            'time',
            ' latitude',
            ' longitude',
        ]


        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # calculations
        self.df[self.columns[' DriveTime (ms)']] *= 1000.
        self.df[self.columns[' StrokeRecoveryTime (ms)']] *= 1000.
        self.df[self.columns[' PeakDriveForce (lbs)']] /= lbstoN
        self.df[self.columns[' DriveLength (meters)']] /= 100.

        wperstroke = self.df['energy_per_stroke']
        fav = wperstroke / self.df[self.columns[' DriveLength (meters)']]
        fav /= lbstoN

        self.df[self.columns[' AverageDriveForce (lbs)']] = fav

        power = self.df[self.columns[' Power (watts)']]
        v = (power / 2.8)**(1. / 3.)
        pace = 500. / v

        self.df[' Stroke500mPace (sec/500m)'] = pace

        seconds = self.df[self.columns['TimeStamp (sec)']]
        res = make_cumvalues(seconds)
        seconds2 = res[0] + seconds[0]
        lapidx = res[1]
        unixtime = seconds2 + totimestamp(self.row_date)

        # get stroke curve
        try:
            data = self.df['curve_data'].str[1:-1].str.split(',',
                                                             expand=True)
            data = data.apply(pd.to_numeric, errors = 'coerce')

            for cols in data.columns.tolist()[1:]:
                data[data<0] = np.nan

            s = []
            for row in data.values.tolist():
                s.append(str(row)[1:-1])

            self.df['curve_data'] = s
        except AttributeError:
            pass
            
        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns['TimeStamp (sec)']] = unixtime
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtime - unixtime.iloc[0]

        self.to_standard()

class MysteryParser(CSVParser):

    def __init__(self, *args, **kwargs):
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        separator = get_separator(1, csvfile)
        kwargs['sep'] = separator
        
        super(MysteryParser, self).__init__(*args, **kwargs)
        self.df = self.df.drop(self.df.index[[0]])
        self.row_date = kwargs.pop('row_date', datetime.datetime.utcnow())

        kwargs['engine'] = 'python'

        

        kwargs['sep'] = None

        self.row_date = kwargs.pop('row_date', datetime.datetime.utcnow())
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

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]
        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # calculations        
        velo = pd.to_numeric(self.df['Speed (m/s)'], errors='coerce')

        pace = 500. / velo
        pace = pace.replace(np.nan, 300)
        pace = pace.replace(np.inf, 300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        seconds = self.df[self.columns['TimeStamp (sec)']]
        res = make_cumvalues_array(np.array(seconds))
        seconds3 = res[0]
        lapidx = res[1]

        spm = self.df[self.columns[' Cadence (stokes/min)']]
        strokelength = velo / (spm / 60.)

        unixtimes = pd.Series(seconds3 + totimestamp(self.row_date))

        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes.iloc[0]
        self.df[self.columns[' StrokeDistance (meters)']] = strokelength

        self.to_standard()

class RowProParser(CSVParser):

    def __init__(self, *args, **kwargs):

        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        separator = get_separator(15, csvfile)

        skipfooter = skip_variable_footer(csvfile)
        kwargs['skipfooter'] = skipfooter
        kwargs['engine'] = 'python'
        kwargs['skiprows'] = 14
        kwargs['usecols'] = None
        kwargs['sep'] = separator

        super(RowProParser, self).__init__(*args, **kwargs)
        self.footer = get_rowpro_footer(csvfile)

        # crude EU format detector
        try:
            p = self.df['Pace'] * 500.
        except TypeError:
            convertlistbase = [
                'Time',
                'Distance',
                'AvgPace',
                'Pace',
                'AvgWatts',
                'Watts',
                'SPM',
                'EndHR'
                ]

            converters = make_converter(convertlistbase,self.df)
            kwargs['converters'] = converters
            super(RowProParser, self).__init__(*args, **kwargs)
            self.footer = get_rowpro_footer(csvfile, converters=converters)

        # replace key values
        footerwork = self.footer[self.footer['Type'] <= 1]
        maxindex = self.df.index[-1]
        endvalue = self.df.loc[maxindex, 'Time']
        #self.df.loc[-1, 'Time'] = 0
        dt = self.df['Time'].diff()
        therowindex = self.df[dt < 0].index

        if len(footerwork) == 2 * (len(therowindex) + 1):
            footerwork = self.footer[self.footer['Type'] == 1]
            self.df.loc[-1, 'Time'] = 0
            dt = self.df['Time'].diff()
            therowindex = self.df[dt < 0].index
            nr = 0
            for i in footerwork.index:
                ttime = footerwork.ix[i, 'Time']
                distance = footerwork.ix[i, 'Distance']
                self.df.ix[therowindex[nr], 'Time'] = ttime
                self.df.ix[therowindex[nr], 'Distance'] = distance
                nr += 1

        if len(footerwork) == len(therowindex) + 1:
            self.df.loc[-1, 'Time'] = 0
            dt = self.df['Time'].diff()
            therowindex = self.df[dt < 0].index
            nr = 0
            for i in footerwork.index:
                ttime = footerwork.ix[i, 'Time']
                distance = footerwork.ix[i, 'Distance']
                self.df.ix[therowindex[nr], 'Time'] = ttime
                self.df.ix[therowindex[nr], 'Distance'] = distance
                nr += 1
        else:
            self.df.loc[maxindex, 'Time'] = endvalue
            for i in footerwork.index:
                ttime = footerwork.ix[i, 'Time']
                distance = footerwork.ix[i, 'Distance']
                diff = self.df['Time'].apply(lambda z: abs(ttime - z))
                diff.sort_values(inplace=True)
                theindex = diff.index[0]
                self.df.ix[theindex, 'Time'] = ttime
                self.df.ix[theindex, 'Distance'] = distance

        dateline = get_file_line(11, csvfile)
        dated = dateline.split(',')[0]
        dated2 = dateline.split(';')[0]
        try:
            self.row_date = parser.parse(dated, fuzzy=True)
        except ValueError:
            self.row_date = parser.parse(dated2, fuzzy=True)

        self.cols = [
            'Time',
            'Distance',
            'SPM',
            'HR',
            'Pace',
            'Watts',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            ' lapIdx',
            ' ElapsedTime (sec)',
            ' latitude',
            ' longitude',
        ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # calculations
        self.df[self.columns[' Stroke500mPace (sec/500m)']] *= 500.0
        seconds = self.df[self.columns['TimeStamp (sec)']] / 1000.
        res = make_cumvalues(seconds)
        seconds2 = res[0] + seconds[0]
        lapidx = res[1]
        seconds3 = seconds2.interpolate()
        seconds3[0] = seconds[0]
        seconds3 = pd.to_timedelta(seconds3, unit='s')
        tts = self.row_date + seconds3

        #unixtimes=tts.apply(lambda x: time.mktime(x.utctimetuple()))
        unixtimes = tts.apply(lambda x: arrow.get(
            x).timestamp + arrow.get(x).microsecond / 1.e6)
        # unixtimes=totimestamp(self.row_date+seconds3)
        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes.iloc[0]

        self.to_standard()

class SpeedCoach2Parser(CSVParser):

    def __init__(self, *args, **kwargs):

        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        skiprows, summaryline, blanklines = skip_variable_header(csvfile)
        unitrow = get_file_line(skiprows + 2, csvfile)
        velo_unit = 'ms'
        dist_unit = 'm'
        if 'KPH' in unitrow:
            velo_unit = 'kph'
        if 'MPH' in unitrow:
            velo_unit = 'mph'

        if 'Kilometer' in unitrow:
            dist_unit = 'km'

        kwargs['skiprows'] = skiprows
        super(SpeedCoach2Parser, self).__init__(*args, **kwargs)
        self.df = self.df.drop(self.df.index[[0]])

        for c in self.df.columns:
            if c not in ['Elapsed Time']:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        self.cols = [
            'Elapsed Time',
            'GPS Distance',
            'Stroke Rate',
            'Heart Rate',
            'Split (GPS)',
            'Power',
            '',
            '',
            '',
            '',
            '',
            'Force Avg',
            'Force Max',
            'Interval',
            ' ElapsedTime (sec)',
            'GPS Lat.',
            'GPS Lon.',
            'GPS Speed',
            'Catch',
            'Slip',
            'Finish',
            'Wash',
            'Work',
            'Max Force Angle',
            'cum_dist',
        ]

        self.defaultcolumnnames += [
            'GPS Speed',
            'catch',
            'slip',
            'finish',
            'wash',
            'driveenergy',
            'peakforceangle',
            'cum_dist',
        ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(zip(self.defaultcolumnnames, self.cols))

        # take Impeller split / speed if available and not zero
        try:
            try:
                impspeed = self.df['Speed (IMP)']
                self.columns['GPS Speed'] = 'Speed (IMP)'
            except KeyError:
                impspeed = self.df['Imp Speed']
                self.columns['GPS Speed'] = 'Imp Speed'
            if impspeed.std() != 0 and impspeed.mean() != 0:
                self.df[self.columns['GPS Speed']] = impspeed
            else:
                self.columns['GPS Speed'] = 'GPS Speed'
        except KeyError:
            pass
        #

        try:
            dist2 = self.df['GPS Distance']
        except KeyError:
            try:
                dist2 = self.df['Distance (GPS)']
                self.columns[' Horizontal (meters)'] = 'Distance (GPS)'
                if 'GPS' in self.columns['GPS Speed']:
                    self.columns['GPS Speed'] = 'Speed (GPS)'
                try:
                    self.df[self.columns[' PeakDriveForce (lbs)']] /= lbstoN
                    self.df[self.columns[' AverageDriveForce (lbs)']] /= lbstoN
                except KeyError:
                    pass
            except KeyError:
                dist2 = self.df['Imp Distance']
                self.columns[' Horizontal (meters)'] = 'Distance (GPS)'
                self.columns[' Stroke500mPace (sec/500m)'] = 'Imp Split'
                self.columns[' Power (watts)'] = 'Work'
                self.columns['Work'] = 'Power'
                self.columns['GPS Speed'] = 'Imp Speed'
                try:
                    self.df[self.columns[' PeakDriveForce (lbs)']] /= lbstoN
                    self.df[self.columns[' AverageDriveForce (lbs)']] /= lbstoN
                except KeyError:
                    pass

        if dist_unit == 'km':
            dist2 *= 1000
            self.df[self.columns[' Horizontal (meters)']] *= 1000.

        cum_dist = make_cumvalues_array(dist2.fillna(method='ffill').values)[0]
        self.df[self.columns['cum_dist']] = cum_dist
        velo = self.df[self.columns['GPS Speed']]
        if velo_unit == 'kph':
            velo = velo / 3.6
        if velo_unit == 'mph':
            velo = velo * 0.44704

        pace = 500. / velo
        pace = pace.replace(np.nan, 300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        # get date from header
        try:
            dateline = get_file_line(4, csvfile)
            dated = dateline.split(',')[1]
            self.row_date = parser.parse(dated, fuzzy=True)
        except ValueError:
            dateline = get_file_line(3, csvfile)
            dated = dateline.split(',')[1]
            self.row_date = parser.parse(dated, fuzzy=True)

        if self.row_date.tzinfo is None or self.row_date.tzinfo.utcoffset(self.row_date) is None:
            try:
                latavg = self.df[self.columns[' latitude']].mean()
                lonavg = self.df[self.columns[' longitude']].mean()
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lng=lonavg, lat=latavg)
                if timezone_str is None:
                    timezone_str = tf.closest_timezone_at(lng=lonavg,
                                                          lat=latavg)
                row_date = self.row_date
                row_date = pytz.timezone(timezone_str).localize(row_date)
            except KeyError:
                row_date = pytz.timezone('UTC').localize(self.row_date)
            self.row_date = row_date

        timestrings = self.df[self.columns['TimeStamp (sec)']]
        seconds = timestrings.apply(
            lambda x: timestrtosecs2(x, unknown=np.nan)
        )
        seconds = clean_nan(np.array(seconds))
        seconds = pd.Series(seconds).fillna(method='ffill').values
        res = make_cumvalues_array(np.array(seconds))
        seconds3 = res[0]
        lapidx = res[1]

        unixtimes = seconds3 + totimestamp(self.row_date)
        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes[0]

        self.to_standard()

        # Read summary data
        skipfooter = 7 + len(self.df)
        if not blanklines:
            skipfooter = skipfooter - 3
        if summaryline:
            self.summarydata = pd.read_csv(csvfile,
                                           skiprows=summaryline,
                                           skipfooter=skipfooter,
                                           engine='python')
            self.summarydata.drop(0, inplace=True)
        else:
            self.summarydata = pd.DataFrame()

    def allstats(self, separator='|'):
        stri = self.summary(separator=separator) + \
            self.intervalstats(separator=separator)
        return stri

    def summary(self, separator='|'):
        stri1 = "Workout Summary - " + self.csvfile + "\n"
        stri1 += "--{sep}Total{sep}-Total-{sep}--Avg--{sep}-Avg-{sep}Avg-{sep}-Avg-{sep}-Max-{sep}-Avg\n".format(
            sep=separator)
        stri1 += "--{sep}Dist-{sep}-Time--{sep}-Pace--{sep}-Pwr-{sep}SPM-{sep}-HR--{sep}-HR--{sep}-DPS\n".format(
            sep=separator)

        d = self.df[self.columns['cum_dist']]
        dist = d.max() - d.min()
        t = self.df[self.columns['TimeStamp (sec)']]
        ttime = t.max() - t.min()
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']].mean()
        try:
            pwr = self.df[self.columns[' Power (watts)']].mean()
        except KeyError:
            pwr = 0

        spm = self.df[self.columns[' Cadence (stokes/min)']].mean()
        try:
            avghr = self.df[self.columns[' HRCur (bpm)']].mean()
            maxhr = self.df[self.columns[' HRCur (bpm)']].max()
        except KeyError:
            avghr = 0
            maxhr = 0

        pacestring = format_pace(pace)
        timestring = format_time(ttime)
        avgdps = self.df['Distance/Stroke (GPS)'].mean()

        stri1 += "--{sep}{dist:0>5.0f}{sep}".format(
            sep=separator,
            dist=dist,
        )

        stri1 += timestring + separator + pacestring

        stri1 += "{sep}{avgpower:0>5.1f}".format(
            sep=separator,
            avgpower=pwr,
        )

        stri1 += "{sep}{avgsr:2.1f}{sep}{avghr:0>5.1f}{sep}".format(
            avgsr=spm,
            sep=separator,
            avghr=avghr
        )

        stri1 += "{maxhr:0>5.1f}{sep}{avgdps:0>4.1f}\n".format(
            sep=separator,
            maxhr=maxhr,
            avgdps=avgdps
        )

        return stri1

    def intervalstats(self, separator='|'):
        stri = "Workout Details\n"
        stri += "#-{sep}SDist{sep}-Split-{sep}-SPace-{sep}-Pwr-{sep}SPM-{sep}AvgHR{sep}DPS-\n".format(
            sep=separator)
        aantal = len(self.summarydata)
        for i in range(aantal):
            sdist = self.summarydata.ix[self.summarydata.index[[i]],
                                        'Total Distance (GPS)']
            split = self.summarydata.ix[self.summarydata.index[[i]],
                                        'Total Elapsed Time']
            space = self.summarydata.ix[self.summarydata.index[[i]],
                                        'Avg Split (GPS)']
            try:
                pwr = self.summarydata.ix[self.summarydata.index[[i]],
                                          'Avg Power']
            except KeyError:
                pwr = 0 * space

            spm = self.summarydata.ix[self.summarydata.index[[i]],
                                      'Avg Stroke Rate']
            try:
                avghr = self.summarydata.ix[self.summarydata.index[[i]],
                                            'Avg Heart Rate']
            except KeyError:
                avghr = 0 * space

            nrstrokes = self.summarydata.ix[self.summarydata.index[[i]],
                                            'Total Strokes']
            dps = float(sdist) / float(nrstrokes)
            splitstring = split.values[0]
            newsplitstring = flexistrftime(flexistrptime(splitstring))
            pacestring = space.values[0]
            newpacestring = flexistrftime(flexistrptime(pacestring))

            stri += "{i:0>2}{sep}{sdist:0>5}{sep}{split}{sep}{space}{sep} {pwr:0>3} {sep}".format(
                i=i + 1,
                sdist=int(float(sdist.values[0])),
                split=newsplitstring,
                space=newpacestring,
                pwr=pwr.values[0],
                sep=separator,
            )
            stri += " {spm} {sep} {avghr:0>3} {sep}{dps:0>4.1f}\n".format(
                sep=separator,
                avghr=avghr.values[0],
                spm=spm.values[0],
                dps=dps,
            )

        return stri
