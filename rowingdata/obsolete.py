from six.moves import range
class OldTCXParser(object):
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
            starttime = arrow.get(startdatetimeobj).timestamp+startdatetimeobj.microsecond/1.e6
        except AttributeError:
            startdatetimeobj = iso8601.parse_date(str(timestamps[0]))
            starttime = arrow.get(startdatetimeobj).timestamp+startdatetimeobj.microsecond/1.e6

        self.activity_starttime = starttime

        # there may be a more elegant and faster way with arrays
        # for i in range(len(timestamps)):
        for time_string in timestamps:
            # time_string = str(timestamps[i])
            time_parsed = iso8601.parse_date(str(time_string))
            unixtimes = np.append(unixtimes,arrow.get(time_parsed).timestamp+time_parsed.microsecond/1.e6)

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
            if spm[i] != 0:
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
            if unixtimes[i+1] != unixtimes[i]:
                velo[i+1] = deltal/(unixtimes[i+1]-unixtimes[i])
            else:
                velo[i+1] = 0

            if spm[i] != 0:
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

        starttime = arrow.get(startdatetimeobj).timestamp+startdatetimeobj.microsecond/1.e6

        self.activity_starttime = starttime

        # there may be a more elegant and faster way with arrays
        #       for i in range(len(timestamps)):
        for time_string in timestamps:
            # time_string = str(timestamps[i])
            time_parsed = iso8601.parse_date(str(time_string))
            unixtimes = np.append(unixtimes,
                                  [arrow.get(time_parsed).timestamp+time_parsed.microsecond/1.0e6])

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
            if spm[i] != 0:
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
            if unixtimes[i+1] != unixtimes[i]:
                velo[i+1] = deltal/(unixtimes[i+1]-unixtimes[i])
            else:
                velo[i+1] = 0

            if spm[i] != 0:
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

