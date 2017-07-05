import time
import iso8601
import numpy as np
import pandas as pd
from pandas import DataFrame
from lxml import objectify
from fitparse import FitFile

from utils import totimestamp, geo_distance, ewmovingaverage

NAMESPACE = 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'

def fitsummarydata(*args, **kwargs):
    from warnings import warn
    warn("fitsummarydata was renamed to FitSummaryData")
    return FitSummaryData(*args,**kwargs)

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
#           if record.mesg_type.name == 'record':
            if record.name == 'record':
                # obtain the values
                speed = record.get_value('speed')
                heartratev = record.get_value('heart_rate')
                spm = record.get_value('cadence')
                power = record.get_value('power')
                if power == None:
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
#           if record.mesg_type.name == 'lap':
            if record.name == 'lap':
                lapcounter += 1

        lat = pd.Series(latlist)
        lon = pd.Series(lonlist)

        velo = pd.Series(velolist)

        pace = 500./velo

        nr_rows = len(lat)

        seconds3 = np.array(timestamp)-timestamp[0]

        self.df = DataFrame({
            'TimeStamp (sec)':timestamp,
            ' Horizontal (meters)': distance,
            ' Cadence (stokes/min)':cadence,
            ' HRCur (bpm)':heartrate,
            ' longitude':lon,
            ' latitude':lat,
            ' Stroke500mPace (sec/500m)':pace,
            ' Power (watts)':powerlist,
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

        # need to select only trackpoints with Cadence, Distance, Time & HR data
        self.selectionstring = '//ns:Trackpoint[descendant::ns:HeartRateBpm]'
        self.selectionstring += '[descendant::ns:Cadence]'
        self.selectionstring += '[descendant::ns:DistanceMeters]'
        self.selectionstring += '[descendant::ns:Time]'


        hr_values = self.root.xpath(self.selectionstring
                                    +'//ns:HeartRateBpm/ns:Value',
                                    namespaces={'ns': NAMESPACE})



        distance_values = self.root.xpath(self.selectionstring
                                          +'/ns:DistanceMeters',
                                          namespaces={'ns': NAMESPACE})

        spm_values = self.root.xpath(self.selectionstring
                                     +'/ns:Cadence',
                                     namespaces={'ns': NAMESPACE})

    def getarray(self, str1, str2=''):
        selectionstring = self.selectionstring
        selectionstring = selectionstring+'//ns:'+str1
        if str2 != '':
            selectionstring = selectionstring+'/ns:'+str2

        the_array = self.root.xpath(selectionstring, namespaces={'ns': NAMESPACE})

        return the_array



class TCXParser(object):
    """ Parser for reading TCX files, e.g. from CrewNerd

    Use: data = rowingdata.TCXParser("crewnerd_data.tcx")

         data.write_csv("crewnerd_data_out.csv")

         """


    def __init__(self, tcx_file):
        tree = objectify.parse(tcx_file)
        self.root = tree.getroot()
        self.data = DataFrame()
        try:
            self.activity = self.root.Activities.Activity
        except AttributeError:
            self.activity = self.root.Courses.Course.Track

        # need to select only trackpoints with Cadence, Distance, Time & HR data
        self.selectionstring = '//ns:Trackpoint[descendant::ns:HeartRateBpm]'
        self.selectionstring += '[descendant::ns:Cadence]'
        self.selectionstring += '[descendant::ns:DistanceMeters]'
        self.selectionstring += '[descendant::ns:Time]'


        hr_values = self.root.xpath(self.selectionstring
                                    +'//ns:HeartRateBpm/ns:Value',
                                    namespaces={'ns': NAMESPACE})



        distance_values = self.root.xpath(self.selectionstring
                                          +'/ns:DistanceMeters',
                                          namespaces={'ns': NAMESPACE})

        spm_values = self.root.xpath(self.selectionstring
                                     +'/ns:Cadence',
                                     namespaces={'ns': NAMESPACE})


        # time stamps (ISO)
        timestamps = self.root.xpath(self.selectionstring
                                     +'/ns:Time',
                                     namespaces={'ns': NAMESPACE})

        lat_values = self.root.xpath(self.selectionstring
                                     +'/ns:Position/ns:LatitudeDegrees',
                                     namespaces={'ns':NAMESPACE})

        long_values = self.root.xpath(self.selectionstring
                                      +'/ns:Position/ns:LongitudeDegrees',
                                      namespaces={'ns':NAMESPACE})

        # Some TCX have no lat/lon
        if not lat_values:
            lat_values = list(0.0*np.array([int(spm) for spm in spm_values]))

        if not long_values:
            long_values = list(0.0*np.array([int(spm) for spm in spm_values]))

        # and here are the trackpoints for "no stroke"
        self.selectionstring2 = '//ns:Trackpoint[descendant::ns:HeartRateBpm]'
        self.selectionstring2 += '[descendant::ns:DistanceMeters]'
        self.selectionstring2 += '[descendant::ns:Time]'

        hr_values2 = self.root.xpath(self.selectionstring2
                                     +'//ns:HeartRateBpm/ns:Value',
                                     namespaces={'ns': NAMESPACE})



        distance_values2 = self.root.xpath(self.selectionstring2
                                           +'/ns:DistanceMeters',
                                           namespaces={'ns': NAMESPACE})

        spm_values2 = np.zeros(len(distance_values2)).tolist()


        # time stamps (ISO)
        timestamps2 = self.root.xpath(self.selectionstring2
                                      +'/ns:Time',
                                      namespaces={'ns': NAMESPACE})

        lat_values2 = self.root.xpath(self.selectionstring2
                                      +'/ns:Position/ns:LatitudeDegrees',
                                      namespaces={'ns':NAMESPACE})

        long_values2 = self.root.xpath(self.selectionstring2
                                       +'/ns:Position/ns:LongitudeDegrees',
                                       namespaces={'ns':NAMESPACE})

        # Some TCX have no lat/lon
        if not lat_values2:
            lat_values2 = list(0.0*np.array([int(spm) for spm in spm_values2]))

        if not long_values2:
            long_values2 = list(0.0*np.array([int(spm) for spm in spm_values2]))

        # merge the two datasets

        timestamps = timestamps+timestamps2

        self.hr_values = hr_values+hr_values2
        self.distance_values = distance_values+distance_values2

        self.spm_values = spm_values+spm_values2

        self.long_values = long_values+long_values2
        self.lat_values = lat_values+lat_values2


        # sort the two datasets
        try:
            data = pd.DataFrame({
                't':timestamps,
                'hr':self.hr_values,
                'd':self.distance_values,
                'spm':self.spm_values,
                'long':self.long_values,
                'lat':self.lat_values
            })
        except ValueError:
            try:
                surro = 0*np.arange(len(timestamps))
                data = pd.DataFrame({
                    't':timestamps,
                    'hr':self.hr_values,
                    'd':self.distance_values,
                    'spm':self.spm_values,
                    'long': surro,
                    'lat': surro
                })
            except ValueError:
                data = pd.DataFrame({
                    't':timestamps,
                    'hr':self.hr_values,
                    'd':self.distance_values,
                    'spm':surro,
                    'long': surro,
                    'lat': surro,
                })




        data = data.drop_duplicates(subset='t')
        data = data.sort_values(by='t', ascending=1)

        timestamps = data.ix[:, 't'].values
        self.hr_values = data.ix[:, 'hr'].values
        self.distance_values = data.ix[:, 'd'].values
        self.spm_values = data.ix[:, 'spm'].values
        self.long_values = data.ix[:, 'long'].values
        self.lat_values = data.ix[:, 'lat'].values


        # convert to unix style time stamp
        unixtimes = np.array([])

        # Activity ID timestamp (start)
        try:
            iso_string = str(self.root.Activities.Activity.Id)
            startdatetimeobj = iso8601.parse_date(iso_string)
            starttime = time.mktime(startdatetimeobj.utctimetuple())+startdatetimeobj.microsecond/1.e6
        except AttributeError:
            startdatetimeobj = iso8601.parse_date(str(timestamps[0]))
            starttime = time.mktime(startdatetimeobj.utctimetuple())+startdatetimeobj.microsecond/1.e6


        self.activity_starttime = starttime

        # there may be a more elegant and faster way with arrays
        # for i in range(len(timestamps)):
        for time_string in timestamps:
            # time_string = str(timestamps[i])
            time_parsed = iso8601.parse_date(str(time_string))
            unixtimes = np.append(unixtimes,
                                  time.mktime(time_parsed.utctimetuple())+time_parsed.microsecond/1.e6)

        self.time_values = unixtimes

        longitude = self.long_values
        lat = self.lat_values
        spm = self.spm_values

        nr_rows = len(lat)
        velo = np.zeros(nr_rows)
        dist2 = np.zeros(nr_rows)
        strokelength = np.zeros(nr_rows)

        for i in range(nr_rows-1):
            res = geo_distance(lat[i], longitude[i], lat[i+1], longitude[i+1])
            deltal = 1000.*res[0]
            dist2[i+1] = dist2[i]+deltal
            velo[i+1] = deltal/(1.0*(unixtimes[i+1]-unixtimes[i]))
            if spm[i] <> 0:
                strokelength[i] = deltal*60/spm[i]
            else:
                strokelength[i] = 0.


        self.strokelength = strokelength
        self.dist2 = dist2
        self.velo = velo

    def write_csv(self, writefile='example.csv', window_size=5, gzip=False):
        # determine if geo data
        lat = pd.Series(self.lat_values)
        lon = pd.Series(self.long_values)

        if lat.std() or lon.std():
            self.write_geo_csv(writefile=writefile,
                               window_size=window_size,
                               gzip=gzip)
        else:
            self.write_nogeo_csv(writefile=writefile,
                                 window_size=window_size,
                                 gzip=gzip)



    def write_geo_csv(self, writefile='example.csv', window_size=5, gzip=False):
        """ Exports TCX data to the CSV format that
        I use in rowingdata
        """

        # Time stamps
        unixtimes = self.time_values

        # Stroke Rate
        spm = self.spm_values

        # Heart Rate
        heartrate = self.hr_values

        longitude = self.long_values
        lat = self.lat_values

        nr_rows = len(spm)
        velo = np.zeros(nr_rows)
        dist2 = np.zeros(nr_rows)
        strokelength = np.zeros(nr_rows)

        velo = self.velo
        strokelength = self.strokelength
        dist2 = self.dist2

        velo2 = ewmovingaverage(velo, window_size)
        strokelength2 = ewmovingaverage(strokelength, window_size)

        pace = 500./velo2
        pace = np.clip(pace, 0, 1e4)


        # Create data frame with all necessary data to write to csv
        data = DataFrame({
            'TimeStamp (sec)':unixtimes,
            ' Horizontal (meters)': dist2,
            ' Cadence (stokes/min)':spm,
            ' HRCur (bpm)':heartrate,
            ' longitude':longitude,
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
            return data.to_csv(writefile+'.gz', index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writefile, index_label='index')



    def write_nogeo_csv(self, writefile='example.csv', window_size=5, gzip=False):
        """ Exports TCX data without position data (indoor)
        to the CSV format that
        I use in rowingdata
        """

        # Time stamps
        unixtimes = self.time_values

        # Distance
        distance = self.distance_values

        # Stroke Rate
        spm = self.spm_values

        # Heart Rate
        heartrate = self.hr_values


        nr_rows = len(spm)
        velo = np.zeros(nr_rows)

        strokelength = np.zeros(nr_rows)

        for i in range(nr_rows-1):
            deltal = distance[i+1]-distance[i]
            if unixtimes[i+1] <> unixtimes[i]:
                velo[i+1] = deltal/(unixtimes[i+1]-unixtimes[i])
            else:
                velo[i+1] = 0

            if spm[i] <> 0:
                strokelength[i] = deltal*60/spm[i]
            else:
                strokelength[i] = 0.


        velo2 = ewmovingaverage(velo, window_size)
        strokelength2 = ewmovingaverage(strokelength, window_size)
        pace = 500./velo2



        # Create data frame with all necessary data to write to csv
        data = DataFrame({
            'TimeStamp (sec)':unixtimes,
            ' Horizontal (meters)': distance,
            ' Cadence (stokes/min)':spm,
            ' HRCur (bpm)':heartrate,
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
            return data.to_csv(writefile+'.gz', index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writefile, index_label='index')



class TCXParserNoHR(object):
    """ Parser for reading TCX files, e.g. from CrewNerd

    Use: data = rowingdata.TCXParser("crewnerd_data.tcx")

         data.write_csv("crewnerd_data_out.csv")

         """


    def __init__(self, tcx_file):
        tree = objectify.parse(tcx_file)
        self.root = tree.getroot()
        self.activity = self.root.Activities.Activity
        self.data = DataFrame()
        # need to select only trackpoints with Cadence, Distance, Time & HR data
        self.selectionstring = '//ns:Trackpoint[descendant::ns:Cadence]'
        self.selectionstring += '[descendant::ns:DistanceMeters]'
        self.selectionstring += '[descendant::ns:Time]'


        distance_values = self.root.xpath(self.selectionstring
                                          +'/ns:DistanceMeters',
                                          namespaces={'ns': NAMESPACE})

        spm_values = self.root.xpath(self.selectionstring
                                     +'/ns:Cadence',
                                     namespaces={'ns': NAMESPACE})


        # time stamps (ISO)
        timestamps = self.root.xpath(self.selectionstring
                                     +'/ns:Time',
                                     namespaces={'ns': NAMESPACE})

        lat_values = self.root.xpath(self.selectionstring
                                     +'/ns:Position/ns:LatitudeDegrees',
                                     namespaces={'ns':NAMESPACE})

        long_values = self.root.xpath(self.selectionstring
                                      +'/ns:Position/ns:LongitudeDegrees',
                                      namespaces={'ns':NAMESPACE})

        # and here are the trackpoints for "no stroke"
        self.selectionstring2 = '//ns:Trackpoint[descendant::ns:DistanceMeters]'
        self.selectionstring2 += '[descendant::ns:Time]'


        distance_values2 = self.root.xpath(self.selectionstring2
                                           +'/ns:DistanceMeters',
                                           namespaces={'ns': NAMESPACE})

        spm_values2 = np.zeros(len(distance_values2)).tolist()


        # time stamps (ISO)
        timestamps2 = self.root.xpath(self.selectionstring2
                                      +'/ns:Time',
                                      namespaces={'ns': NAMESPACE})

        lat_values2 = self.root.xpath(self.selectionstring2
                                      +'/ns:Position/ns:LatitudeDegrees',
                                      namespaces={'ns':NAMESPACE})

        long_values2 = self.root.xpath(self.selectionstring2
                                       +'/ns:Position/ns:LongitudeDegrees',
                                       namespaces={'ns':NAMESPACE})

        # merge the two datasets

        timestamps = timestamps+timestamps2

        self.distance_values = distance_values+distance_values2

        self.spm_values = spm_values+spm_values2

        self.long_values = long_values+long_values2
        self.lat_values = lat_values+lat_values2

        # correct lat, long if empty
        if not len(self.lat_values):
            self.lat_values = np.zeros(len(self.spm_values))
            self.long_values = np.zeros(len(self.spm_values))
            

        # sort the two datasets
        data = pd.DataFrame({
            't':timestamps,
            'd':self.distance_values,
            'spm':self.spm_values,
            'long':self.long_values,
            'lat':self.lat_values
            })

        data = data.drop_duplicates(subset='t')
        data = data.sort_values(by='t', ascending=1)

        timestamps = data.ix[:, 't'].values
        self.distance_values = data.ix[:, 'd'].values
        self.spm_values = data.ix[:, 'spm'].values
        self.long_values = data.ix[:, 'long'].values
        self.lat_values = data.ix[:, 'lat'].values

        # convert to unix style time stamp
        unixtimes = np.array([])

        # Activity ID timestamp (start)
        iso_string = str(self.root.Activities.Activity.Id)
        startdatetimeobj = iso8601.parse_date(iso_string)

        starttime = time.mktime(startdatetimeobj.utctimetuple())+startdatetimeobj.microsecond/1.e6

        self.activity_starttime = starttime

        # there may be a more elegant and faster way with arrays
        #       for i in range(len(timestamps)):
        for time_string in timestamps:
            # time_string = str(timestamps[i])
            time_parsed = iso8601.parse_date(str(time_string))
            unixtimes = np.append(unixtimes,
                                  [time.mktime(time_parsed.utctimetuple())+time_parsed.microsecond/1.e6])

        self.time_values = unixtimes

        longitude = self.long_values
        lat = self.lat_values
        spm = self.spm_values

        nr_rows = len(lat)
        velo = np.zeros(nr_rows)
        dist2 = np.zeros(nr_rows)
        strokelength = np.zeros(nr_rows)

        for i in range(nr_rows-1):
            res = geo_distance(lat[i], longitude[i], lat[i+1], longitude[i+1])
            deltal = 1000.*res[0]
            dist2[i+1] = dist2[i]+deltal
            velo[i+1] = deltal/(1.0*(unixtimes[i+1]-unixtimes[i]))
            if spm[i] <> 0:
                strokelength[i] = deltal*60/spm[i]
            else:
                strokelength[i] = 0.


        self.strokelength = strokelength
        self.dist2 = dist2
        self.velo = velo


    def write_csv(self, writefile='example.csv', window_size=5, gzip=False):
        # determine if geo data
        lat = pd.Series(self.lat_values)
        lon = pd.Series(self.long_values)

        if lat.std() or lon.std():
            self.write_geo_csv(writefile=writefile,
                               window_size=window_size,
                               gzip=gzip)
        else:
            self.write_nogeo_csv(writefile=writefile,
                                 window_size=window_size,
                                 gzip=gzip)


    def write_geo_csv(self, writefile='example.csv', window_size=5, gzip=False):
        """ Exports TCX data to the CSV format that
        I use in rowingdata
        """

        # Time stamps
        unixtimes = self.time_values


        # Stroke Rate
        spm = self.spm_values

        # Heart Rate

        longitude = self.long_values
        lat = self.lat_values

        nr_rows = len(spm)
        velo = np.zeros(nr_rows)
        dist2 = np.zeros(nr_rows)
        strokelength = np.zeros(nr_rows)

        velo = self.velo
        strokelength = self.strokelength
        dist2 = self.dist2

        velo2 = ewmovingaverage(velo, window_size)
        strokelength2 = ewmovingaverage(strokelength, window_size)

        pace = 500./velo2
        pace = np.clip(pace, 0, 1e4)



        # Create data frame with all necessary data to write to csv
        data = DataFrame({
            'TimeStamp (sec)':unixtimes,
            ' Horizontal (meters)': dist2,
            ' Cadence (stokes/min)':spm,
            ' HRCur (bpm)':np.zeros(nr_rows),
            ' longitude':longitude,
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

            return data.to_csv(writefile+'.gz', index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writefile, index_label='index')



    def write_nogeo_csv(self, writefile='example.csv', window_size=5, gzip=False):
        """ Exports TCX data without position data (indoor)
        to the CSV format that
        I use in rowingdata
        """

        # Time stamps
        unixtimes = self.time_values


        # Distance Meters
        distance = self.distance_values

        # Stroke Rate
        spm = self.spm_values


        nr_rows = len(spm)
        velo = np.zeros(nr_rows)

        strokelength = np.zeros(nr_rows)

        for i in range(nr_rows-1):
            deltal = distance[i+1]-distance[i]
            if unixtimes[i+1] <> unixtimes[i]:
                velo[i+1] = deltal/(unixtimes[i+1]-unixtimes[i])
            else:
                velo[i+1] = 0

            if spm[i] <> 0:
                strokelength[i] = deltal*60/spm[i]
            else:
                strokelength[i] = 0.


        velo2 = ewmovingaverage(velo, window_size)
        strokelength2 = ewmovingaverage(strokelength, window_size)
        pace = 500./velo2



        # Create data frame with all necessary data to write to csv
        data = DataFrame({
            'TimeStamp (sec)':unixtimes,
            ' Horizontal (meters)': distance,
            ' Cadence (stokes/min)':spm,
            ' HRCur (bpm)':np.zeros(nr_rows),
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
            ' ElapsedTime (sec)':unixtimes-self.activity_starttime,
        })

        if gzip:
            return data.to_csv(writefile+'.gz',
                               index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writefile, index_label='index')

