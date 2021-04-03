# pylint: disable=C0103, C0303
from __future__ import absolute_import
#from builtins import (bytes, str, open, super, range,
#                      zip, round, input, int, pow, object)
import os
import io
import csv
import gzip
import zipfile
import re
import datetime
import pytz
import arrow
import iso8601
import traceback
import shutil
import numpy as np
import pandas as pd
import codecs
from pandas.core.indexing import IndexingError

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from pandas import Series, DataFrame
from dateutil import parser

from timezonefinder import TimezoneFinder
from lxml import objectify,etree
from fitparse import FitFile

try:
    from .utils import (
        totimestamp, format_pace, format_time,
    )

    from .tcxtools import strip_control_characters
except (ValueError,ImportError):
    from rowingdata.utils import (
        totimestamp, format_pace, format_time,
    )

    from rowingdata.tcxtools import strip_control_characters


import six
from six.moves import range
from six.moves import zip

import sys
if sys.version_info[0]<=2:
    pythonversion = 2
    readmode = 'r'
    readmodebin = 'rb'
else:
    readmode = 'rt'
    readmodebin = 'rt'
    pythonversion = 3
    from io import open


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
                try:
                    t = datetime.datetime.strptime(inttime, "%M:%S.%f")
                except ValueError:
                    t = datetime.datetime.utcnow()

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

def csvtests(s):
    # get first and 7th line of file

    try:
        firstline = s[0]
    except IndexError:
        firstline = ''

    try:
        secondline = s[1]
    except IndexError:
        secondline = ''

    try:
        thirdline = s[2]
    except IndexError:
        thirdline = ''

    try:
        fourthline = s[3]
    except IndexError:
        fourthline = ''

    try:
        seventhline = s[6]
    except IndexError:
        seventhline = ''

    try:
        ninthline = s[8]
    except IndexError:
        ninthline = ''

    if 'timestamp' in firstline and 'InstaSpeed' in firstline:
        return 'nklinklogbook'


    if 'RitmoTime' in firstline:
        return 'ritmotime'

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

    if 'Avg Speed (IMP)' in firstline:
        return 'speedcoach2'

    if 'LiNK' in ninthline:
        return 'speedcoach2'

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

    if 'Club' in firstline and 'Piece Stroke Count' in secondline:
        return 'boatcoachotw'

    if 'peak_force_pos' in firstline:
        return 'rowperfect3'

    if 'Hair' in seventhline:
        return 'rp'

    if 'smo2' in thirdline:
        return 'humon'

    if 'Total elapsed time (s)' in firstline:
        return 'ergstick'
    if 'Total elapsed time' in firstline:
        return 'ergstick'

    if 'Stroke Number' and 'Time (seconds)' in firstline:
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

    if '500m Split (secs)' in firstline and 'Force Curve Data Points (Newtons)' in firstline:
        return 'eth' # it's unknown but it was first submitted by a student from ETH Zurich

    return 'unknown'

def get_file_type(f):
    filename,extension = os.path.splitext(f)
    extension = extension.lower()

    if extension == '.xls':
        return 'xls'
    if extension == '.kml':
        return 'kml'
    if extension == '.txt':
        if os.path.basename(f)[0:3].lower() == 'att':
            return 'att'
    if extension == '.gz':
        filename,extension = os.path.splitext(filename)
        if extension == '.fit':
            newfile = 'temp.fit'
            with gzip.open(f,'rb') as fop:
                with open(newfile,'wb') as f_out:
                    shutil.copyfileobj(fop, f_out)

                try:
                    FitFile(newfile, check_crc=False).parse()
                    return 'fit'
                except:
                    return 'unknown'

            return 'fit'
        if extension == '.tcx':
            try:
                tree = etree.parse(f)
                root = tree.getroot()
                return 'tcx'
            except:
                try:
                    with open(path, 'r') as fop:
                        input = fop.read()
                        input = strip_control_characters(input)
                    with open('temp_xml.tcx','w') as fout:
                        fout.write(input)

                    tree = etree.parse('temp_xml.tcx')
                    os.remove('temp_xml.tcx')
                    return 'tcx'
                except:
                    return 'unknown'
        if extension == '.gpx':
            try:
                tree = etree.parse(f)
                root = tree.getroot()
                return 'gpx'
            except:
                return 'unknown'

        with gzip.open(f, readmode) as fop:
            try:
                if extension == '.csv':
                    s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')
                    return csvtests(s)

            except IOError:
                return 'notgzip'
    if extension == '.csv':
        linecount,isbinary = get_file_linecount(f)
        if linecount <= 2:
            return 'nostrokes'

        if isbinary:
            with open(f,readmodebin) as fop:
                s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')
        else:
            with open(f, readmode) as fop:
                s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')

        return csvtests(s)

    if extension == '.tcx':
        try:
            tree = etree.parse(f)
            root = tree.getroot()
            return 'tcx'
        except:
            try:
                with open(f, 'r') as fop:
                    input = fop.read()
                    input = strip_control_characters(input)
                with open('temp_xml.tcx','w') as ftemp:
                    ftemp.write(input)

                tree = etree.parse('temp_xml.tcx')
                os.remove('temp_xml.tcx')
                return 'tcx'
            except:
                return 'unknown'
    if extension == '.gpx':
        try:
            tree = etree.parse(f)
            root = tree.getroot()
            return 'gpx'
        except:
            return 'unknown'

    if extension == '.fit':
        try:
            FitFile(f, check_crc=False).parse()
        except:
            return 'unknown'

        return 'fit'

    if extension == '.zip':
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
    #    extension = f[-3:].lower()
    extension = os.path.splitext(f)[1].lower()
    isbinary = False
    if extension == '.gz':
        with gzip.open(f,'rb') as fop:
            count = sum(1 for line in fop if line.rstrip('\n'))
        if count <= 2:
            # test for \r
            with gzip.open(f,readmodebin) as fop:
                s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')
                count = len(s)

    else:
        with open(f, 'r') as fop:
            try:
                count = sum(1 for line in fop if line.rstrip('\n'))
            except:
                return 0,False
        if count <= 2:
            # test for \r
            with open(f,readmodebin) as fop:
                isbinary = True
                s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')
                count = len(s)

    return count,isbinary

def get_file_line(linenr, f, isbinary=False):
    line = ''
    extension = os.path.splitext(f)[1].lower()
    # extension = f[-3:].lower()
    if extension == '.gz':
        with gzip.open(f, readmode) as fop:
            s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')
    else:
        with open(f, readmodebin) as fop:
            s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')

    return s[linenr-1]


def get_separator(linenr, f):
    line = ''
    extension = os.path.splitext(f)[1].lower()
    # extension = f[-3:].lower()
    if extension == '.gz':
        with gzip.open(f, readmode) as fop:
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

def empower_bug_correction(oarlength,inboard,a,b):
    f = (oarlength-inboard-b)/(oarlength-inboard-a)

    return f

def getoarlength(line):
    l = float(line.split(',')[-1])

    return l

def getinboard(line):
    inboard = float(line.split(',')[-1])
    return inboard

def getfirmware(line):
    l = line.lower().split(',')
    try:
        firmware = l[l.index("firmware version:")+1]
    except ValueError:
        firmware = ''

    return firmware

def get_empower_rigging(f):
    oarlength = 289.
    inboard = 88.
    line = '1'
    try:
        with open(f, readmode) as fop:
            for line in fop:
                if 'Oar Length' in line:
                    try:
                        oarlength = getoarlength(line)
                    except ValueError:
                        return None,None
                if 'Inboard' in line:
                    try:
                        inboard = getinboard(line)
                    except ValueError:
                        return None,None
    except (UnicodeDecodeError,ValueError):
        with gzip.open(f, readmode) as fop:
            for line in fop:
                if 'Oar Length' in line:
                    try:
                        oarlength = getoarlength(line)
                    except ValueError:
                        return None,None
                if 'Inboard' in line:
                    try:
                        inboard = getinboard(line)
                    except ValueError:
                        return None,None



    return oarlength / 100., inboard / 100.

def get_empower_firmware(f):
    firmware = ''
    try:
        with open(f,readmode) as fop:
            for line in fop:
                if 'firmware' in line.lower() and 'oar' in line.lower():
                    firmware = getfirmware(line)
    except (IndexError,UnicodeDecodeError):
        with gzip.open(f,readmode) as fop:
            for line in fop:
                if 'firmware' in line.lower() and 'oar' in line.lower():
                    firmware = getfirmware(line)



    try:
        firmware = np.float(firmware)
    except ValueError:
        firmware = None

    return firmware

def skip_variable_footer(f):
    counter = 0
    counter2 = 0

    extension = os.path.splitext(f)[1].lower()
    # extension = f[-3:].lower()
    if extension == '.gz':
        fop = gzip.open(f,readmode)
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


    extension = os.path.splitext(f)[1].lower()
    # extension = f[-3:].lower()
    if extension == '.gz':
        fop = gzip.open(f,readmode)
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
    sessionc = -2
    summaryc = -2
    extension = os.path.splitext(f)[1].lower()
    # extension = f[-3:].lower()
    firmware = ''
    if extension == '.gz':
        with gzip.open(f,readmode) as fop:
            s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')
    else:
        with open(f,readmodebin) as fop:
            s = fop.read().replace('\r\n','\n').replace('\r','\n').split('\n')


    summaryfound = False
    for line in s:
        if line.startswith('Session Summary'):
            sessionc = counter
            summaryfound = True
        if line.startswith('Interval Summaries'):
            summaryc = counter
        if 'firmware' in line.lower() and 'oar' in line.lower():
            firmware = getfirmware(line)
        if line.startswith('Session Detail Data') or line.startswith('Per-Stroke Data'):
            counter2 = counter
        else:
            counter += 1



    # test for blank line
    l = s[counter2+1]
    # l = get_file_line(counter2 + 2, f)
    if 'Interval' in l:
        counter2 = counter2 - 1
        summaryc = summaryc - 1
        blanklines = 0
    else:
        blanklines = 1

    return counter2 + 2, summaryc + 2, blanklines, sessionc + 2

def ritmo_variable_header(f):
    counter = 0
    extension = os.path.splitext(f)[1].lower()
    # extension = f[-3:].lower()
    if extension == '.gz':
        fop = gzip.open(f,readmode)
    else:
        fop = open(f, 'r')

    for line in fop:
        if line.startswith('#'):
            counter += 1
        else:
            fop.close()
            return counter

    return counter

def bc_variable_header(f):
    counter = 0
    extension = os.path.splitext(f)[1].lower()
    # extension = f[-3:].lower()
    if extension == '.gz':
        fop = gzip.open(f,readmode)
    else:
        fop = open(f, 'r')


    for line in fop:
        if line.startswith('Workout'):
            fop.close()
            return counter+1
        else:
            counter += 1

    fop.close()

    return 0

def make_cumvalues_array(xvalues,doequal=False):
    """ Takes a Pandas dataframe with one column as input value.
    Tries to create a cumulative series.

    """

    try:
        newvalues = 0.0 * xvalues
    except TypeError:
        return [xvalues,0]

    dx = np.diff(xvalues)
    dxpos = dx
    nrsteps = len(dxpos[dxpos < 0])
    lapidx = np.append(0, np.cumsum((-dx + abs(dx)) / (-2 * dx)))
    if doequal:
        lapidx[0] = 0
        cntr = 0
        for i in range(len(dx)-1):
            if dx[i+1] <= 0:
                cntr += 1
            lapidx[i+1] = cntr
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
        newvalues = np.cumsum(dxpos) + xvalues.iloc[0]
        newvalues.iloc[0] = xvalues.iloc[0]
    else:
        newvalues = xvalues

    newvalues = newvalues.replace([-np.inf, np.inf], np.nan)


    newvalues.fillna(method='ffill', inplace=True)
    newvalues.fillna(method='bfill', inplace=True)
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
        skipfooter = kwargs.pop('skipfooter', 0)
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

        try:
            x = self.df['TimeStamp (sec)']
        except KeyError:
            cols = self.df.columns
            for col in cols:
                if 'TimeStamp ' in col:
                    self.df['TimeStamp (sec)'] = self.df[col]

        self.columns = {c: c for c in self.defaultcolumnnames}

    def to_standard(self):
        inverted = {value: key for key, value in six.iteritems(self.columns)}
        self.df.rename(columns=inverted, inplace=True)
        self.columns = {c: c for c in self.defaultcolumnnames}

    def time_values(self, *args, **kwargs):
        timecolumn = kwargs.pop('timecolumn', 'TimeStamp (sec)')
        unixtimes = self.df[timecolumn]

        return unixtimes

    def write_csv(self, *args, **kwargs):
        if self.df.empty:
            return None

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

# Parsing ETH files
class ETHParser(CSVParser):
    def __init__(self, *args, **kwargs):
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        super(ETHParser, self).__init__(*args, **kwargs)

        self.cols = [
            '',
            'Distance (meters)',
            'Stroke Rate (s/m)',
            'Heart Rate (bpm)',
            '500m Split (secs)',
            'Power (Watts)',
            'Drive Length (meters)',
            '',
            'Drive Time (secs)',
            '',
            '',
            'Average Drive Force (Newtons)',
            'Peak Force (Newtons)',
            '',
            'Time (secs)',
            '',
            '',
        ]

        self.cols = [b if a == '' else a
                     for a,b in zip(self.cols, self.defaultcolumnnames)]
        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # calculations
        self.df[self.columns[' DriveTime (ms)']] *= 1000.

        startdatetime = datetime.datetime.utcnow()
        elapsed = self.df[self.columns[' ElapsedTime (sec)']]
        starttimeunix = arrow.get(startdatetime).timestamp()

        unixtimes = starttimeunix+elapsed
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes-starttimeunix

        self.to_standard()


# Parsing CSV files from Humon
class HumonParser(CSVParser):
    def __init__(self, *args, **kwargs):
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        skiprows = 2
        kwargs['skiprows'] = skiprows

        super(HumonParser, self).__init__(*args, **kwargs)

        self.cols = [
            'Time [seconds]',
            'distance [meters]',
            '',
            'heartRate [bpm]',
            'speed [meters/sec]',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'Time [seconds]',
            'latitude [degrees]',
            'longitude [degrees]',
        ]


        self.cols = [b if a == '' else a
                     for a,b in zip(self.cols, self.defaultcolumnnames)]
        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # calculations
        velo = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        pace = 500./velo
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        # get date from header
        dateline = get_file_line(2,csvfile)
        row_datetime = parser.parse(dateline, fuzzy=True,yearfirst=True,dayfirst=False)

        timestamp = arrow.get(row_datetime).timestamp()

        time = self.df[self.columns['TimeStamp (sec)']]
        time += timestamp
        self.df[self.columns['TimeStamp (sec)']] = time

        self.to_standard()

class RitmoTimeParser(CSVParser):
    def __init__(self, *args, **kwargs):
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        skiprows = ritmo_variable_header(csvfile)
        kwargs['skiprows'] = skiprows

        separator = get_separator(skiprows+2, csvfile)
        kwargs['sep'] = separator

        super(RitmoTimeParser, self).__init__(*args, **kwargs)
        # crude EU format detector
        try:
            ll = self.df['Longitude (deg)']*10.0
        except TypeError:
            convertlistbase = [
                'Total Time (sec)',
                'Rate (spm)',
                'Distance (m)',
                'Speed (m/s)',
                'Latitude (deg)',
                'Longitude (deg)',
            ]
            converters = make_converter(convertlistbase,self.df)

            kwargs['converters'] = converters

            super(RitmoTimeParser, self).__init__(*args, **kwargs)

        self.cols = [
            '',
            'Distance (m)',
            'Rate (spm)',
            'Heart Rate (bpm)',
            'Split (/500m)',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'Piece#',
            'Total Time (sec)',
            'Latitude (deg)',
            'Longitude (deg)',
        ]

        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # calculations / speed
        velo = self.df['Speed (m/s)']
        pace = 500. / velo
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        # add rest from state column
        self.df[' WorkoutState'] = self.df['State'].apply(lambda x: 3 if x.lower()=='rest' else 4)

        # try date from first line
        firstline = get_file_line(1, csvfile)
        try:
            startdatetime = parser.parse(firstline,fuzzy=True)
        except ValueError:
            startdatetime = datetime.datetime.utcnow()

        if startdatetime.tzinfo is None:
            try:
                latavg = self.df[self.columns[' latitude']].mean()
                lonavg = self.df[self.columns[' longitude']].mean()
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lng=lonavg, lat=latavg)
                if timezone_str is None:
                    timezone_str = tf.closest_timezone_at(lng=lonavg,lat=latavg)

                startdatetime = pytz.timezone(timezone_str).localize(startdatetime)
            except KeyError:
                startdatetime = pytz.timezone('UTC').localize(startdatetime)
                timezonestr = 'UTC'

        elapsed = self.df[self.columns[' ElapsedTime (sec)']]
        starttimeunix = arrow.get(startdatetime).timestamp()
        #tts = startdatetime + elapsed.apply(lambda x: datetime.timedelta(seconds=x))
        #unixtimes=tts.apply(lambda x: time.mktime(x.utctimetuple()))
        #unixtimes = tts.apply(lambda x: arrow.get(
        #    x).timestamp() + arrow.get(x).microsecond / 1.e6)
        unixtimes = starttimeunix+elapsed
        self.df[self.columns['TimeStamp (sec)']] = unixtimes

        self.to_standard()

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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

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
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        skiprows = bc_variable_header(csvfile)
        kwargs['skiprows'] = skiprows

        separator = get_separator(3, csvfile)
        kwargs['sep'] = separator

        super(BoatCoachOTWParser, self).__init__(*args, **kwargs)

        # 500m or km based
        try:
            pace = self.df['Last 10 Stroke Speed(/500m)']
        except KeyError:
            pace1 = self.df['Last 10 Stroke Speed(/km)']
            self.df['Last 10 Stroke Speed(/500m)'] = pace1.values


        # crude EU format detector
        try:
            ll = self.df['Longitude']*10.0
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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        try:
            row_datetime = self.df[self.columns['TimeStamp (sec)']]
            row_date = parser.parse(row_datetime[0], fuzzy=True,yearfirst=True,dayfirst=False)
            row_datetime = row_datetime.apply(
                lambda x: parser.parse(x, fuzzy=True,yearfirst=True,dayfirst=False))
            unixtimes = row_datetime.apply(lambda x: arrow.get(
                x).timestamp() + arrow.get(x).microsecond / 1.e6)
        except KeyError:
            row_date2 = arrow.get(row_date).timestamp()
            timecolumn = self.df[self.columns[' ElapsedTime (sec)']]
            timesecs = timecolumn.apply(timestrtosecs)
            timesecs = make_cumvalues(timesecs)[0]
            unixtimes = row_date2 + timesecs

        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'

        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes[0]

        try:
            d = self.df['Last 10 Stroke Speed(/km)']
            multiplicator = 0.5
        except:
            multiplicator = 1


        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']].apply(
                timestrtosecs2
            )

        pace *= multiplicator


        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace


        self.to_standard()

class CoxMateParser(CSVParser):

    def __init__(self, *args, **kwargs):
        super(CoxMateParser, self).__init__(*args, **kwargs)
        # remove "00 waiting to row"

        self.cols = [
            '',
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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

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
            x).timestamp() + arrow.get(x).microsecond / 1.e6)
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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # calculations
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']] / 2.
        pace = np.clip(pace, 0, 1e4)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace
        timestamps = self.df[self.columns['TimeStamp (sec)']]
        # convert to unix style time stamp
        tts = timestamps.apply(lambda x: iso8601.parse_date(x[2:-1]))
        #unixtimes=tts.apply(lambda x: time.mktime(x.utctimetuple()))
        unixtimes = tts.apply(lambda x: arrow.get(x).timestamp() + arrow.get(x).microsecond / 1.e6)
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[
            self.columns[' ElapsedTime (sec)']
        ] = unixtimes - unixtimes.iloc[0]
        self.to_standard()


class BoatCoachParser(CSVParser):

    def __init__(self, *args, **kwargs):
        kwargs['skiprows'] = 1
        kwargs['usecols'] = list(range(25))

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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # get date from footer
        try:
            try:
                with open(csvfile, readmode) as fop:
                    line = fop.readline()
                    dated = re.split('Date:', line)[1][1:-1]
            except (IndexError,UnicodeDecodeError):
                with gzip.open(csvfile,readmode) as fop:
                    line = fop.readline()
                    dated = re.split('Date:', line)[1][1:-1]
            row_date = parser.parse(dated, fuzzy=True,yearfirst=True,dayfirst=False)
        except IOError:
            pass


        try:
            datetime = self.df[self.columns['TimeStamp (sec)']]
            row_date = parser.parse(datetime[0], fuzzy=True,yearfirst=True,dayfirst=False)
            row_datetime = datetime.apply(lambda x: parser.parse(x, fuzzy=True,yearfirst=True,dayfirst=False))
            unixtimes = row_datetime.apply(
                lambda x: arrow.get(x).timestamp() + arrow.get(x).microsecond / 1.e6
            )
        except KeyError:
            # calculations
            # row_date2=time.mktime(row_date.utctimetuple())
            row_date2 = arrow.get(row_date).timestamp()
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

        moving = self.df[self.columns[' Horizontal (meters)']].diff()
        moving = moving.apply(lambda x:abs(x))

        power[dif < 5] = self.df[self.columns[' Power (watts)']][dif < 5]

        power[dif > 1000] = self.df[self.columns[' Power (watts)']][dif > 1000]

        power[moving <= 1] = 0

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

            with gzip.open(csvfile,readmode) as f:
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
        while len(self.df.loc[mask]) > 2:
            mask = (self.df['cumdist'] == maxdist)
            self.df.drop(self.df.index[-1], inplace=True)

        mask = (self.df['cumdist'] == maxdist)
        try:
            self.df.loc[
                mask,
                self.columns[' lapIdx']
            ] = self.df.loc[self.df.index[-3], self.columns[' lapIdx']]
        except IndexError:
            pass


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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        row_datetime = self.df[self.columns['TimeStamp (sec)']]
        row_datetime = row_datetime.apply(
            lambda x: parser.parse(x, fuzzy=True,yearfirst=True,dayfirst=False))
        #unixtimes=datetime.apply(lambda x: time.mktime(x.utctimetuple()))
        unixtimes = row_datetime.apply(lambda x: arrow.get(
            x).timestamp() + arrow.get(x).microsecond / 1.e6)
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
        kwargs['usecols'] = list(range(25))

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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # get date from footer
        try:
            with open(csvfile, 'r') as fop:
                line = fop.readline()
                dated = re.split('Date:', line)[1][1:-1]
        except (IndexError,UnicodeDecodeError):
            with gzip.open(csvfile,readmode) as fop:
                line = fop.readline()
                dated = re.split('Date:', line)[1][1:-1]

        row_date = parser.parse(dated, fuzzy=True,yearfirst=True,dayfirst=False)

        try:
            row_datetime = self.df[self.columns['TimeStamp (sec)']]
            row_date = parser.parse(datetime[0], fuzzy=True,yearfirst=True,dayfirst=False)
            rowdatetime = row_datetime.apply(lambda x: parser.parse(x, fuzzy=True,yearfirst=True,dayfirst=False))
            unixtimes = row_datetime.apply(lambda x: arrow.get(
                x).timestamp() + arrow.get(x).microsecond / 1.e6)
        except KeyError:
            # calculations
            # row_date2=time.mktime(row_date.utctimetuple())
            row_date2 = arrow.get(row_date).timestamp()
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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # calculations
        # get date from footer
        pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
        try:
            pace = np.clip(pace, 0, 1e4)
            pace = pace.replace(0, 300)
        except TypeError:
            pass
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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # calculations
        try:
            self.df[self.columns[' DriveTime (ms)']] *= 1000.
            self.df[self.columns[' StrokeRecoveryTime (ms)']] *= 1000.
        except KeyError:
            pass

        try:
            pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
            pace = np.clip(pace, 1, 1e4)
            self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace
        except TypeError:
            pace = self.df[self.columns[' Stroke500mPace (sec/500m)']]
            pace = pace.apply(lambda x:flexistrptime(x))
            pace = pace.apply(lambda x:60*x.minute+x.second+x.microsecond/1.e6)
            pace = np.clip(pace, 1, 1e4)
            self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        # check distance
        try:
            distance = self.df[self.columns[' Horizontal (meters)']]
        except KeyError:
            self.columns[' Horizontal (meters)'] = 'Total distance'
            distance = self.df[self.columns[' Horizontal (meters)']]
            distance = distance.apply(lambda x:int(x[:-2]))
            self.df[self.columns[' Horizontal (meters)']] = distance

        #velocity = 500. / pace
        #power = 2.8 * velocity**3

        #self.df[' Power (watts)'] = power

        try:
            seconds = self.df[self.columns['TimeStamp (sec)']]
        except:
            self.columns['TimeStamp (sec)'] = 'Total elapsed time'
            seconds = self.df[self.columns['TimeStamp (sec)']]
            seconds = seconds.apply(lambda x:flexistrptime(x))
            seconds = seconds.apply(lambda x:3600*x.hour+60*x.minute+x.second+x.microsecond/1.e6)

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
        except AttributeError as e:
            pass
            # print traceback.format_exc()

        for c in self.df.columns:
            if c != 'curve_data':
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
            'work_per_pulse'
        ]

        self.defaultcolumnnames += [
            'driveenergy'
        ]


        self.cols = [b if a == '' else a
                     for a, b in zip(self.cols, self.defaultcolumnnames)]

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))


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
        newseconds,lapidx = make_cumvalues_array(seconds)
        newstrokenr,lapidx = make_cumvalues_array(self.df['stroke_number'],doequal=True)
        seconds2 = pd.Series(newseconds)+newseconds[0]
        res = make_cumvalues(seconds)
        #seconds2 = res[0] + seconds[0]
        #lapidx = res[1]
        unixtime = seconds2 + totimestamp(self.row_date)


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
        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

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

        # HR versions
        try:
            hr = self.df[self.columns[' HRCur (bpm)']]
        except KeyError:
            hr = self.df['HR (BPM)']
            self.df[self.columns[' HRCur (bpm)']] = hr


        spm = self.df[self.columns[' Cadence (stokes/min)']]
        try:
            strokelength = velo / (spm / 60.)
        except TypeError:
            strokelength = 0*velo

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
                ttime = footerwork.loc[i, 'Time']
                distance = footerwork.loc[i, 'Distance']
                self.df.loc[therowindex[nr], 'Time'] = ttime
                self.df.loc[therowindex[nr], 'Distance'] = distance
                nr += 1

        if len(footerwork) == len(therowindex) + 1:
            self.df.loc[-1, 'Time'] = 0
            dt = self.df['Time'].diff()
            therowindex = self.df[dt < 0].index
            nr = 0
            for i in footerwork.index:
                ttime = footerwork.loc[i, 'Time']
                distance = footerwork.loc[i, 'Distance']
                self.df.loc[therowindex[nr], 'Time'] = ttime
                self.df.loc[therowindex[nr], 'Distance'] = distance
                nr += 1
        else:
            self.df.loc[maxindex, 'Time'] = endvalue
            for i in footerwork.index:
                ttime = footerwork.loc[i, 'Time']
                distance = footerwork.loc[i, 'Distance']
                diff = self.df['Time'].apply(lambda z: abs(ttime - z))
                diff.sort_values(inplace=True)
                theindex = diff.index[0]
                self.df.loc[theindex, 'Time'] = ttime
                self.df.loc[theindex, 'Distance'] = distance

        dateline = get_file_line(11, csvfile)
        dated = dateline.split(',')[0]
        dated2 = dateline.split(';')[0]
        try:
            self.row_date = parser.parse(dated, fuzzy=True,yearfirst=True,dayfirst=False)
        except ValueError:
            self.row_date = parser.parse(dated2, fuzzy=True,yearfirst=True,dayfirst=False)

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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # calculations
        self.df[self.columns[' Stroke500mPace (sec/500m)']] *= 500.0
        seconds = self.df[self.columns['TimeStamp (sec)']] / 1000.
        res = make_cumvalues(seconds)
        seconds2 = res[0] + seconds[0]
        lapidx = res[1]
        seconds3 = seconds2.interpolate()
        seconds3[0] = seconds[0]
        unixtimes = seconds3 + arrow.get(self.row_date).timestamp()

#        seconds3 = pd.to_timedelta(seconds3, unit='s')


#        try:
#            tts = self.row_date + seconds3
#            unixtimes = tts.apply(lambda x: arrow.get(
#                x).timestamp() + arrow.get(x).microsecond / 1.e6)
#        except ValueError:
#            seconds3 = seconds2.interpolate()



        self.df[self.columns[' lapIdx']] = lapidx
        self.df[self.columns['TimeStamp (sec)']] = unixtimes
        self.columns[' ElapsedTime (sec)'] = ' ElapsedTime (sec)'
        self.df[self.columns[' ElapsedTime (sec)']] = unixtimes - unixtimes.iloc[0]

        self.to_standard()

class NKLiNKLogbookParser(CSVParser):
    def __init__(self, *args, **kwargs):
        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        super(NKLiNKLogbookParser, self).__init__(*args, **kwargs)

        self.cols = [
            'timestamp',
            'gpsTotalDistance',
            'strokeRate',
            'heartRate',
            'gpsPace',
            'power',
            '',
            'gpsDistStroke',
            'driveTime',
            '',
            '',
            'handleForceAvg',
            'maxHandleForce',
            'sessionIntervalId',
            'elapsedTime',
            'latitude',
            'longitude',
            'gpsInstaSpeed',
            'catchAngle',
            'slip',
            'finishAngle',
            'wash',
            'realWorkPerStroke',
            'positionOfMaxForce',
            'gpsTotalDistance',
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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # do something with impeller stuff

        # force is in Newtons
        self.df[self.columns[' PeakDriveForce (lbs)']] /= lbstoN
        self.df[self.columns[' AverageDriveForce (lbs)']] /= lbstoN

        # timestamp is in milliseconds
        self.df[self.columns['TimeStamp (sec)']] /= 1000.
        self.df[self.columns[' ElapsedTime (sec)']] /= 1000.

        self.df[' StrokeRecoveryTime (ms)'] = self.df['cycleTime']-self.df[self.columns[' DriveTime (ms)']]


        self.to_standard()

        self.df = self.df.sort_values(by='TimeStamp (sec)',ascending=True)




class SpeedCoach2Parser(CSVParser):

    def __init__(self, *args, **kwargs):

        if args:
            csvfile = args[0]
        else:
            csvfile = kwargs['csvfile']

        skiprows, summaryline, blanklines, sessionline = skip_variable_header(csvfile)

        firmware = get_empower_firmware(csvfile)
        corr_factor = 1.0
        if firmware is not None:
            if firmware < 2.18:
                # apply correction
                oarlength, inboard = get_empower_rigging(csvfile)
                if oarlength is not None and oarlength > 3.30:
                    # sweep
                    a = 0.15
                    b = 0.275
                    corr_factor = empower_bug_correction(oarlength,inboard,a,b)
                elif oarlength is not None and oarlength <= 3.3:
                    # scull
                    a = 0.06
                    b = 0.225
                    corr_factor = empower_bug_correction(oarlength,inboard,a,b)



        unitrow = get_file_line(skiprows + 2, csvfile)
        self.velo_unit = 'ms'
        self.dist_unit = 'm'
        if 'KPH' in unitrow:
            self.velo_unit = 'kph'
        if 'MPH' in unitrow:
            self.velo_unit = 'mph'

        if 'Kilometer' in unitrow:
            self.dist_unit = 'km'

        if 'Newtons' in unitrow:
            self.force_unit = 'N'
        else:
            self.force_unit = 'lbs'

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

        self.columns = dict(list(zip(self.defaultcolumnnames, self.cols)))

        # correct Power, Work per Stroke
        try:
            self.df[self.columns[' Power (watts)']] *= corr_factor
            self.df[self.columns['driveenergy']] *= corr_factor
        except KeyError:
            pass

        # set GPS speed apart for swapping
        try:
            self.df['GPSSpeed'] = self.df['GPS Speed']
            self.df['GPSDistance'] = self.df['GPS Distance']
        except KeyError:
            try:
                self.df['GPSSpeed'] = self.df['Speed (GPS)']
                self.df['GPSDistance'] = self.df['Distance (GPS)']
                self.columns['GPS Speed'] = 'Speed (GPS)'
                self.columns[' Horizontal (meters)'] = 'Distance (GPS)'
            except KeyError:
                pass

        # take Impeller split / speed if available and not zero
        try:
            impspeed = self.df['Speed (IMP)']
            self.columns['GPS Speed'] = 'Speed (IMP)'
            self.columns[' Horizontal (meters)'] = 'Distance (IMP)'
            self.df['ImpellerSpeed'] = impspeed
            self.df['ImpellerDistance'] = self.df['Distance (IMP)']
        except KeyError:
            try:
                impspeed = self.df['Imp Speed']
                self.columns['GPS Speed'] = 'Imp Speed'
                self.columns[' Horizontal (meters)'] = 'Imp Distance'
                self.df['ImpellerSpeed'] = impspeed
                self.df['ImpellerDistance'] = self.df['Imp Distance']
            except KeyError:
                impspeed = 0*self.df[self.columns['GPS Speed']]

        if impspeed.std() != 0 and impspeed.mean() != 0:
            self.df[self.columns['GPS Speed']] = impspeed
        else:
            self.columns['GPS Speed'] = 'GPS Speed'
            self.columns[' Horizontal (meters)'] = 'GPS Distance'

        #

        try:
            dist2 = self.df[self.columns[' Horizontal (meters)']]
        except KeyError:
            try:
                dist2 = self.df['Distance (GPS)']
                self.columns[' Horizontal (meters)'] = 'Distance (GPS)'
                if 'GPS' in self.columns['GPS Speed']:
                    self.columns['GPS Speed'] = 'Speed (GPS)'
            except KeyError:
                try:
                    dist2 = self.df['Imp Distance']
                    self.columns[' Horizontal (meters)'] = 'Distance (GPS)'
                    self.columns[' Stroke500mPace (sec/500m)'] = 'Imp Split'
                    self.columns[' Power (watts)'] = 'Work'
                    self.columns['Work'] = 'Power'
                    self.columns['GPS Speed'] = 'Imp Speed'
                except KeyError:
                    dist2 = self.df['Distance (IMP)']
                    self.columns[' Stroke500mPace (sec/500m)'] = 'Split (IMP)'
                    self.columns[' Horizontal (meters)'] = 'Distance (GPS)'
                    self.columns[' Power (watts)'] = 'Work'
                    self.columns['Work'] = 'Power'
                    self.columns['GPS Speed'] = 'Speed (IMP)'

        try:
            if self.force_unit == 'N':
                self.df[self.columns[' PeakDriveForce (lbs)']] /= lbstoN
                self.df[self.columns[' AverageDriveForce (lbs)']] /= lbstoN
        except KeyError:
            pass

        if self.dist_unit == 'km':
            #dist2 *= 1000
            self.df[self.columns[' Horizontal (meters)']] *= 1000.
            try:
                self.df['GPSDistance'] *= 1000.
            except KeyError:
                pass
            try:
                self.df['ImpellerDistance'] *= 1000.
            except KeyError:
                pass

        cum_dist = make_cumvalues_array(dist2.fillna(method='ffill').values)[0]
        self.df[self.columns['cum_dist']] = cum_dist
        velo = self.df[self.columns['GPS Speed']]
        if self.velo_unit == 'kph':
            velo = velo / 3.6
        if self.velo_unit == 'mph':
            velo = velo * 0.44704

        pace = 500. / velo
        pace = pace.replace(np.nan, 300)
        self.df[self.columns[' Stroke500mPace (sec/500m)']] = pace

        # get date from header
        try:
            dateline = get_file_line(4, csvfile)
            dated = dateline.split(',')[1]
            # self.row_date = parser.parse(dated, fuzzy=True, dayfirst=False)
            self.row_date = parser.parse(dated,fuzzy=False,dayfirst=False)
            alt_date = parser.parse(dated,fuzzy=False,dayfirst=True)
        except ValueError:
            dateline = get_file_line(3, csvfile)
            dated = dateline.split(',')[1]
            try:
                #                self.row_date = parser.parse(dated, fuzzy=True,dayfirst=False)
                self.row_date = parser.parse(dated, fuzzy=False,dayfirst=False)
                alt_date = parser.parse(dated,fuzzy=False,dayfirst=True)
            except ValueError:
                self.row_date = datetime.datetime.now()
                alt_date = self.row_date

        if alt_date.month == datetime.datetime.now().month:
            if alt_date != self.row_date:
                self.row_date = alt_date


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

        if not self.df.empty:
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
            try:
                self.summarydata = pd.read_csv(csvfile,
                                               skiprows=summaryline,
                                               skipfooter=skipfooter,
                                               engine='python')
                self.summarydata.drop(0, inplace=True)
            except:
                self.summarydata = pd.DataFrame()
        else:
            self.summarydata = pd.DataFrame()

        skipfooter = 11 + len(self.df)+len(self.summarydata)
        if not blanklines:
            skipfooter = skipfooter - 3
        if sessionline:
            try:
                self.sessiondata = pd.read_csv(csvfile,
                                               skiprows= sessionline,
                                               skipfooter=skipfooter,
                                               engine='python')
                self.sessiondata.drop(0,inplace=True)
            except:
                self.sessiondata = pd.DataFrame()
        else:
            self.sessiondata = pd.DataFrame()

    def impellerconsistent(self, threshold = 0.3):
        impellerconsistent = True
        try:
            impspeed = self.df['ImpellerSpeed']
        except KeyError:
            return False, True, 0

        nrvalues = len(impspeed)

        impspeed.fillna(inplace=True,value=0)
        nrvalid = impspeed.astype(bool).sum()

        ratio = float(nrvalues-nrvalid)/float(nrvalues)

        if ratio > threshold:
            impellerconsistent = False

        return True, impellerconsistent, ratio

    def allstats(self, separator='|'):
        stri = self.summary(separator=separator) + \
            self.intervalstats(separator=separator)
        return stri

    def sessionsummary(self, separator= '|'):
        if self.sessiondata.empty:
            return None

        stri1 = "Workout Summary - " + self.csvfile + "\n"
        stri1 += "--{sep}Total{sep}-Total-{sep}--Avg--{sep}-Avg-{sep}-Avg--{sep}-Avg-{sep}-Max-{sep}-Avg\n".format(
            sep=separator)
        stri1 += "--{sep}Dist-{sep}-Time--{sep}-Pace--{sep}-Pwr-{sep}-SPM--{sep}-HR--{sep}-HR--{sep}-DPS\n".format(
            sep=separator)

        try:
            dist = self.sessiondata['Total Distance (GPS)'].astype(float).mean()
        except (KeyError,ValueError):
            try:
                dist = self.sessiondata['Total Distance'].astype(float).mean()
            except (ValueError,KeyError):
                dist = 0.0


        timestring = self.sessiondata['Total Elapsed Time'].values[0]
        timestring = flexistrftime(flexistrptime(timestring))

        try:
            pacestring = self.sessiondata['Avg Split (GPS)'].values[0]
        except KeyError:
            pacestring = self.sessiondata['Avg Split'].values[0]

        pacestring = flexistrftime(flexistrptime(pacestring))
        try:
            pwr = self.sessiondata['Avg Power'].astype(float).mean()
        except (KeyError,ValueError):
            pwr = 0.0


        try:
            spm = self.sessiondata['Avg Stroke Rate'].astype(float).mean()
        except (ValueError,KeyError):
            spm = 0

        try:
            avghr = self.sessiondata['Avg Heart Rate'].astype(float).mean()
        except (ValueError,KeyError):
            avghr = 0

        try:
            avgdps = self.sessiondata['Distance/Stroke (GPS)'].astype(float).mean()
        except KeyError:
            avgdps = 0

        try:
            maxhr = self.df[self.columns[' HRCur (bpm)']].max()
        except KeyError:
            maxhr = 0

        stri1 += "--{sep}{dist:0>5.0f}{sep}".format(
            sep=separator,
            dist=dist,
        )

        stri1 += timestring + separator + pacestring

        stri1 += "{sep}{avgpower:0>5.1f}".format(
            sep=separator,
            avgpower=pwr,
        )

        stri1 += "{sep} {avgsr:2.1f} {sep}{avghr:0>5.1f}{sep}".format(
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


    def summary(self, separator='|'):
        if self.sessionsummary() is not None:
            return self.sessionsummary()

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
        stri += "#-{sep}SDist{sep}-Split-{sep}-SPace-{sep}-Pwr--{sep}-SPM--{sep}AvgHR{sep}DPS-\n".format(
            sep=separator)
        aantal = len(self.summarydata)
        for i in self.summarydata.index:
            sdist = self.summarydata.loc[i,
                                        'Total Distance (GPS)']

            if self.dist_unit == 'km':
                sdist = float(sdist)*1000.

            split = self.summarydata.loc[i,
                                        'Total Elapsed Time']
            space = self.summarydata.loc[i,
                                        'Avg Split (GPS)']
            try:
                pwr = self.summarydata.loc[i,
                                          'Avg Power']
            except KeyError:
                pwr = 0 * space

            spm = self.summarydata.loc[i,
                                      'Avg Stroke Rate']
            try:
                avghr = self.summarydata.loc[i,
                                            'Avg Heart Rate']
            except KeyError:
                avghr = 0 * space

            nrstrokes = self.summarydata.loc[i,
                                            'Total Strokes']
            try:
                dps = float(sdist) / float(nrstrokes)
            except ZeroDivisionError:
                dps = 0.0

            splitstring = split


            newsplitstring = flexistrftime(flexistrptime(splitstring))
            pacestring = space
            newpacestring = flexistrftime(flexistrptime(pacestring))

            stri += "{i:0>2}{sep}{sdist:0>5}{sep}{split}{sep}{space}{sep} {pwr:0>3} {sep}".format(
                i=i + 1,
                sdist=int(float(sdist)),
                split=newsplitstring,
                space=newpacestring,
                pwr=pwr,
                sep=separator,
            )
            stri += " {spm} {sep} {avghr:0>3} {sep}{dps:0>4.1f}\n".format(
                sep=separator,
                avghr=avghr,
                spm=spm,
                dps=dps,
            )

        return stri
