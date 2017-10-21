# pylint: disable=C0103, C0303
import numpy as np
import pandas as pd
from pandas import DataFrame
from lxml import objectify
from fitparse import FitFile
import tcxtools
from utils import totimestamp, geo_distance

NAMESPACE = 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'

def tofloat(x):
    try:
        return float(x)
    except ValueError:
        return np.nan
    

def fitsummarydata(*args, **kwargs):
    from warnings import warn
    warn("fitsummarydata was renamed to FitSummaryData")
    return FitSummaryData(*args, **kwargs)

class FitSummaryData(object):
    def __init__(self, readfile):
        self.readfile = readfile
        self.fitfile = FitFile(readfile, check_crc=False)
        self.records = self.fitfile.messages
        self.summarytext = 'Work Details\n'


    def setsummary(self, separator="|"):
        lapcount = 0
        self.summarytext += "#-{sep}SDist{sep}-Split-{sep}-SPace-{sep}-SPM-{sep}-Pwr-{sep}AvgHR{sep}MaxHR{sep}DPS-\n".format(
            sep=separator
            )

        strokecount = 0
        recordcount = 0
        totalhr = 0
        totalpower = 0
        maxhr = 0

        totaldistance = 0
        totaltime = 0
        grandhr = 0
        grandmaxhr = 0
        grandpower = 0

        for record in self.records:
            if record.name == 'record':
                heartrate = record.get_value('heart_rate')
                if heartrate is None:
                    heartrate = 0
                if heartrate > maxhr:
                    maxhr = heartrate

                if heartrate > grandmaxhr:
                    grandmaxhr = heartrate

                power = record.get_value('power')
                if power is None:
                    power = 0

                totalhr += heartrate
                grandhr += heartrate

                totalpower += power
                grandpower += power
                
                strokecount += 1
                recordcount += 1

            if record.name == 'lap':
                lapcount += 1

                try:
                    inthr = int(totalhr/float(strokecount))
                except ZeroDivisionError:
                    inthr = 0

                try:
                    intpower = int(totalpower/float(strokecount))
                except ZeroDivisionError:
                    intpower = 0
                    
                inttime = record.get_value('total_elapsed_time')

                lapmin = int(inttime/60)
                lapsec = int(int(10*(inttime-lapmin*60.))/10.)

                intdist = int(record.get_value('total_distance'))
                try:
                    intvelo = intdist/inttime
                except ZeroDivisionError:
                    intvelo = 1.0
                    
                intpace = 500./intvelo

                totaldistance += intdist
                totaltime += inttime

                try:
                    intspm = 60.*strokecount/inttime
                except ZeroDivisionError:
                    intspm = 1.0

                try:
                    intdps = intdist/float(strokecount)
                except ZeroDivisionError:
                    intdps = 0.0

                intmaxhr = maxhr

                pacemin = int(intpace/60)
                pacesec = int(10*(intpace-pacemin*60.))/10.
                pacestring = str(pacemin)+":"+str(pacesec)

                strokecount = 0
                totalhr = 0
                totalpower = 0
                maxhr = 0


                summarystring = "{nr:0>2}{sep}{intdist:0>5d}{sep}".format(
                    nr=lapcount,
                    sep=separator,
                    intdist=intdist
                    )

                summarystring += " {lapmin:0>2}:{lapsec:0>2} {sep}".format(
                    lapmin=lapmin,
                    lapsec=lapsec,
                    sep=separator,
                    )

                summarystring += "{pacemin:0>2}:{pacesec:0>3.1f}".format(
                    pacemin=pacemin,
                    pacesec=pacesec,
                    )

                summarystring += "{sep} {intspm:0>4.1f}{sep}".format(
                    intspm=intspm,
                    sep=separator
                    )

                summarystring += " {intpower:0>3d} {sep}".format(
                    intpower=intpower,
                    sep=separator
                    )
                
                summarystring += " {inthr:0>3d} {sep}".format(
                    inthr=inthr,
                    sep=separator
                    )

                summarystring += " {intmaxhr:0>3d} {sep}".format(
                    intmaxhr=intmaxhr,
                    sep=separator
                    )

                summarystring += " {dps:0>3.1f}".format(
                    dps=intdps
                    )

                summarystring += "\n"
                self.summarytext += summarystring

        # add total summary
        try:
            overallvelo = totaldistance/totaltime
        except ZeroDivisionError:
            overallvelo = 1.0
            
        overallpace = 500./overallvelo


        minutes = int(overallpace/60)
        sec = int(10*(overallpace-minutes*60.))/10.
        pacestring = str(minutes)+":"+str(sec)

        totmin = int(totaltime/60)
        totsec = int(int(10*(totaltime-totmin*60.))/10.)

        avghr = grandhr/float(recordcount)
        avgpower = grandpower/float(recordcount)
        try:
            avgspm = 60.*recordcount/totaltime
        except ZeroDivisionError:
            avgspm = 1.0

        avgdps = totaldistance/float(recordcount)

        summarystring = "Workout Summary\n"
        summarystring += "--{sep}{totaldistance:0>5}{sep}".format(
            totaldistance=int(totaldistance),
            sep=separator
            )

        summarystring += " {totmin:0>2}:{totsec:0>2} {sep} ".format(
            totmin=totmin,
            totsec=totsec,
            sep=separator,
            )

        summarystring += pacestring+separator

        summarystring += " {avgspm:0>4.1f}{sep}".format(
            sep=separator,
            avgspm=avgspm
            )

        summarystring += " {avgpower:0>3} {sep}".format(
            sep=separator,
            avgpower=int(avgpower)
            )

        summarystring += " {avghr:0>3} {sep} {grandmaxhr:0>3} {sep}".format(
            avghr=int(avghr),
            grandmaxhr=int(grandmaxhr),
            sep=separator
            )

        summarystring += " {avgdps:0>3.1f}".format(
            avgdps=avgdps
            )

        self.summarytext += summarystring

class FITParser(object):

    def __init__(self, readfile):
        self.readfile = readfile
        self.fitfile = FitFile(readfile, check_crc=False)
        self.records = self.fitfile.messages
        cadence = []
        heartrate = []
        latlist = []
        lonlist = []
        powerlist = []
        velolist = []
        timestamp = []
        distance = []
        lapidx = []
        lapcounter = 0

        for record in self.records:
            if record.name == 'record':
                # obtain the values
                speed = record.get_value('speed')
                heartratev = record.get_value('heart_rate')
                spm = record.get_value('cadence')
                power = record.get_value('power')
                if power is None:
                    power = 0
                distancev = record.get_value('distance')
                timestampv = record.get_value('timestamp')
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
                    velolist.append(speed)
                    heartrate.append(heartratev)
                    lapidx.append(lapcounter)
                    latlist.append(latv)
                    lonlist.append(lonv)
                    timestamp.append(totimestamp(timestampv))
                    cadence.append(spm)
                    distance.append(distancev)
                    powerlist.append(power)
            if record.name == 'lap':
                lapcounter += 1

        lat = pd.Series(latlist)
        lon = pd.Series(lonlist)

        velo = pd.Series(velolist)

        pace = 500./velo

        nr_rows = len(lat)

        seconds3 = np.array(timestamp)-timestamp[0]

        self.df = DataFrame({
            'TimeStamp (sec)': timestamp,
            ' Horizontal (meters)': distance,
            ' Cadence (stokes/min)': cadence,
            ' HRCur (bpm)': heartrate,
            ' longitude': lon,
            ' latitude': lat,
            ' Stroke500mPace (sec/500m)': pace,
            ' Power (watts)': powerlist,
            ' DriveLength (meters)': np.zeros(nr_rows),
            ' StrokeDistance (meters)': np.zeros(nr_rows),
            ' DriveTime (ms)': np.zeros(nr_rows),
            ' DragFactor': np.zeros(nr_rows),
            ' StrokeRecoveryTime (ms)': np.zeros(nr_rows),
            ' AverageDriveForce (lbs)': np.zeros(nr_rows),
            ' PeakDriveForce (lbs)': np.zeros(nr_rows),
            ' ElapsedTime (sec)': seconds3,
            ' lapIdx': lapidx,
            })

    def write_csv(self, writefile="fit_o.csv", gzip=False):

        if gzip:
            return self.df.to_csv(writefile+'.gz', index_label='index',
                                  compression='gzip')
        else:
            return self.df.to_csv(writefile, index_label='index')


class TCXParserTester(object):
    def __init__(self, tcx_file):
        tree = objectify.parse(tcx_file)
        self.root = tree.getroot()
        self.activity = self.root.Activities.Activity

        # need to select only trackpoints with Cadence, Distance,
        # Time & HR data
        self.selectionstring = '//ns:Trackpoint[descendant::ns:HeartRateBpm]'
        self.selectionstring += '[descendant::ns:Cadence]'
        self.selectionstring += '[descendant::ns:DistanceMeters]'
        self.selectionstring += '[descendant::ns:Time]'


        self.hr_values = self.root.xpath(self.selectionstring
                                         +'//ns:HeartRateBpm/ns:Value',
                                         namespaces={'ns': NAMESPACE})



        self.distance_values = self.root.xpath(self.selectionstring
                                               +'/ns:DistanceMeters',
                                               namespaces={'ns': NAMESPACE})

        self.spm_values = self.root.xpath(self.selectionstring
                                          +'/ns:Cadence',
                                          namespaces={'ns': NAMESPACE})

    def getarray(self, str1, str2=''):
        selectionstring = self.selectionstring
        selectionstring = selectionstring+'//ns:'+str1
        if str2 != '':
            selectionstring = selectionstring+'/ns:'+str2

        the_array = self.root.xpath(selectionstring,
                                    namespaces={'ns': NAMESPACE})

        return the_array


class TCXParser(object):
    def __init__(self, tcx_file):
        self.df = tcxtools.tcxtodf(tcx_file)
        lat = self.df['latitude'].apply(tofloat).values
        longitude = self.df['longitude'].apply(tofloat).values
        unixtimes = self.df['timestamp'].values
        try:
            spm = self.df['Cadence'].apply(tofloat).values
        except KeyError:
            spm = self.df['StrokeRate'].apply(tofloat).values
            self.df['Cadence'] = self.df['StrokeRate']

        try:
            velo = self.df['Speed'].apply(tofloat)
            dist2 = self.df['DistanceMeters'].apply(tofloat)
            strokelength = velo*60./spm
        except KeyError:
            nr_rows = len(lat)
            dist2 = np.zeros(nr_rows)
            velo = np.zeros(nr_rows)
            strokelength = np.zeros(nr_rows)
            for i in range(nr_rows-1):
                res = geo_distance(lat[i], longitude[i], lat[i+1], longitude[i+1])
                deltal = 1000.*res[0]
                dist2[i+1] = dist2[i]+deltal
                try:
                    velo[i+1] = deltal/(1.0*(unixtimes[i+1]-unixtimes[i]))
                except ZeroDivisionError:
                    velo[i+1] = velo[i]
                if spm[i] <> 0:
                    strokelength[i] = deltal*60/spm[i]
                else:
                    strokelength[i] = 0.
        
        try:
            power = self.df['Watts']
        except KeyError:
            self.df['Watts'] = 0*spm

        p = 500./velo

        self.df[' Horizontal (meters)'] = dist2
        self.df[' StrokeDistance (meters)'] = strokelength
        self.df[' Stroke500mPace (sec/500m)'] = p

        # translate from standard TCX names to our naming convention
        self.columns = {
            'timestamp':'TimeStamp (sec)',
            'Cadence': ' Cadence (stokes/min)',
            'HeartRateBpm' : ' HRCur (bpm)',
            'Watts': ' Power (watts)',
            'lapid': ' lapIdx',
            'latitude': ' latitude',
            'longitude': ' longitude',
        }

        self.df.rename(columns=self.columns, inplace=True)

        cc = [value for key, value in self.columns.items()]

        for c in cc:
            if c != 'lapIdx':
                try:
                    self.df[c] = self.df[c].astype(float)
                except KeyError:
                    pass
                    

    def write_csv(self, writefile='example.csv', window_size=5, gzip=False):
        data = self.df
        data = data.sort_values(by='TimeStamp (sec)', ascending=True)
        data = data.fillna(method='ffill')

        # drop all-zero columns
        for c in data.columns:
            if (data[c] == 0).any() and data[c].mean() == 0:
                data = data.drop(c, axis=1)
            if c == 'Position':
                data = data.drop(c, axis=1)
            if c == 'Extensions':
                data = data.drop(c, axis=1)
                        
        if gzip:
            return data.to_csv(writefile+'.gz', index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writefile, index_label='index')
