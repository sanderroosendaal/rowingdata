from __future__ import absolute_import
from __future__ import print_function
import time
import datetime
from dateutil import parser as ps
import lxml
import arrow
import numpy as np
from lxml import etree, objectify
from lxml.etree import XMLSyntaxError
import six.moves.urllib.request, six.moves.urllib.error, six.moves.urllib.parse
from six.moves import range

import sys
if sys.version_info[0]<=2:
    pythonversion = 2
    textwritemode = 'w'
else:
    pythonversion = 3
    textwritemode = 'wt'
    from io import open


namespace='http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'

empty_gpx = """
<?xml version="1.0" encoding="UTF-8" standalone="no" ?><gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxx="http://www.garmin.com/xmlschemas/GpxExtensions/v3" xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1" creator="Oregon 400t" version="1.1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd http://www.garmin.com/xmlschemas/GpxExtensions/v3 http://www.garmin.com/xmlschemas/GpxExtensionsv3.xsd http://www.garmin.com/xmlschemas/TrackPointExtension/v1 http://www.garmin.com/xmlschemas/TrackPointExtensionv1.xsd"><metadata><link href="http://www.garmin.com"><text>Garmin International</text></link><time>2018-03-17T12:59:13</time></metadata><trk><name>Export by rowingdata</name>
</trk></gpx>
"""

def lap_begin(f,datetimestring,totalmeters,avghr,maxhr,avgspm,totalseconds):
    f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>')
    f.write('<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxx="http://www.garmin.com/xmlschemas/GpxExtensions/v3" xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1" creator="Oregon 400t" version="1.1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd http://www.garmin.com/xmlschemas/GpxExtensions/v3 http://www.garmin.com/xmlschemas/GpxExtensionsv3.xsd http://www.garmin.com/xmlschemas/TrackPointExtension/v1 http://www.garmin.com/xmlschemas/TrackPointExtensionv1.xsd"><metadata><link href="http://www.garmin.com"><text>Garmin International</text></link>')


def write_gpx(gpxFile,df,row_date="2016-01-01",notes="Exported by rowingdata"):
    if notes==None:
        notes="Exported by rowingdata"
    f=open(gpxFile,'w')

    totalseconds=int(df['TimeStamp (sec)'].max()-df['TimeStamp (sec)'].min())
    totalmeters=int(df['cum_dist'].max())
    avghr=int(df[' HRCur (bpm)'].mean())
    if avghr == 0:
        avghr=1
    maxhr=int(df[' HRCur (bpm)'].max())
    if maxhr == 0:
        maxhr=1
    avgspm=int(df[' Cadence (stokes/min)'].mean())

    seconds=df['TimeStamp (sec)'].values
    distancemeters=df['cum_dist'].values
    heartrate=df[' HRCur (bpm)'].values.astype(int)
    cadence=np.round(df[' Cadence (stokes/min)'].values).astype(int)

    nr_rows=len(seconds)

    try:
        lat=df[' latitude'].values
    except KeyError:
        lat=np.zeros(nr_rows)

    try:
        lon=df[' longitude'].values
    except KeyError:
        lon=np.zeros(nr_rows)

    haspower=1

    try:
        power=df[' Power (watts)'].values
    except KeyError:
        haspower=0

    s="2000-01-01"
    tt=ps.parse(s)
    #timezero=time.mktime(tt.timetuple())
    timezero = arrow.get(tt).timestamp()
    if seconds[0]<timezero:
        # print("Taking Row_Date ",row_date)
        dateobj=ps.parse(row_date)
        #unixtimes=seconds+time.mktime(dateobj.timetuple())
        unixtimes=seconds+arrow.get(dateobj).timestamp()



    datetimestring=row_date

    lap_begin(f,datetimestring,totalmeters,avghr,maxhr,avgspm,totalseconds)

    ts = datetime.datetime.fromtimestamp(unixtimes[0]).isoformat()
    s = '<time>{ts}</time></metadata><trk><name>Export by rowingdata</name><trkseg>'.format(
        ts=ts,
        )

    f.write(s)


    for i in range(nr_rows):
        s = '          <trkpt lat="{lat}" lon="{lon}">\n'.format(
            lat=lat[i],
            lon=lon[i]
            )
        f.write(s)
        #s=datetime.datetime.fromtimestamp(unixtimes[i]).isoformat()
        s = arrow.get(unixtimes[i]).isoformat()
        f.write('            <time>{s}</time>\n'.format(s=s))
        f.write('          </trkpt>\n')



    f.write('</trkseg>')
    f.write('</trk>')
    f.write('</gpx>')

    f.close()

    file=open(gpxFile,'r')

    some_xml_string=file.read()

    file.close()

    try:
        xsd_file=six.moves.urllib.request.urlopen("http://www.topografix.com/GPX/1/1/gpx.xsd")
        output=open('gpx.xsd','w')
        if pythonversion <= 2:
            output.write(xsd_file.read().replace('\n',''))
        else:
            output.write(xsd_file.read().decode('utf-8').replace('\n',''))
        output.close()
        xsd_filename="gpx.xsd"

        # Run some tests
        try:
            tree=objectify.parse(gpxFile)
            try:
                schema=etree.XMLSchema(file=xsd_filename)
                parser=objectify.makeparser(schema=schema)
                objectify.fromstring(some_xml_string, parser)
                # print("YEAH!, your xml file has validated")
            except XMLSyntaxError:

                print("Oh NO!, your xml file does not validate")
                pass
        except:
            print("Oh No!, your xml file does not validate")
            pass

    except six.moves.urllib.error.URLError:
        print("cannot download GPX schema")
        print("your GPX file is unvalidated. Good luck")




    return 1
