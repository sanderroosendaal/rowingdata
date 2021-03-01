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
import ssl
from six.moves import range
import xml.etree.ElementTree as et

from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

import sys
if sys.version_info[0]<=2:
    pythonversion = 2
    textwritemode = 'w'
else:
    pythonversion = 3
    textwritemode = 'wt'
    from io import open

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")



def get_empty_tcx():
    top = Element('TrainingCenterDatabase')
    top.attrib['xmlns'] = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
    activities = SubElement(top,'Activities')
    activity = SubElement(activities,'Activity')
    activity.attrib['Sport'] = "Other"
    id = SubElement(activity,'Id')
    id.text = '2015-03-28T20:45:15.000Z'
    creator = SubElement(activity,'Creator')
    creator.attrib['xsi:type'] = 'Device_t'
    creator.attrib['xmlns:xsi'] = 'http://www.w3.org/2001/XMLSchema-instance'
    name = SubElement(creator,'Name')
    name.text = 'Empty File'
    unitid = SubElement(creator,'UnitId')
    unitid.text = '0'
    productid = SubElement(creator,'ProductID')
    productid.text = '0'

    return prettify(top)


namespace='http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'

def lap_begin(f,datetimestring,totalmeters,avghr,maxhr,avgspm,totalseconds):
    f.write('        <Lap StartTime="{s}">\n'.format(s=datetimestring))
    f.write('          <TotalTimeSeconds>{s}</TotalTimeSeconds>\n'.format(s=totalseconds))
    f.write('          <DistanceMeters>{s}</DistanceMeters>\n'.format(s=totalmeters))
    f.write('          <Calories>1</Calories>\n')
    f.write('          <AverageHeartRateBpm xsi:type="HeartRateInBeatsPerMinute_t">\n')
    f.write('            <Value>{s}</Value>\n'.format(s=avghr))
    f.write('          </AverageHeartRateBpm>\n')

    f.write('          <MaximumHeartRateBpm xsi:type="HeartRateInBeatsPerMinute_t">\n')
    f.write('            <Value>{s}</Value>\n'.format(s=maxhr))
    f.write('          </MaximumHeartRateBpm>\n')
    f.write('          <Intensity>Active</Intensity>\n')
    f.write('          <Cadence>{s}</Cadence>\n'.format(s=avgspm))
    f.write('          <TriggerMethod>Manual</TriggerMethod>\n')
    f.write('          <Track>\n')


def lap_end(f):
    f.write('          </Track>\n')
    f.write('          <Notes>rowingdata export</Notes>\n')
    f.write('        </Lap>\n')

def create_tcx(df,row_date="2016-01-01", notes="Exported by rowingdata",
               sport="Other"):
    if notes is None:
        notes="Exported by rowingdata"

    notes = notes.encode('utf-8')

    try:
        totalseconds=int(df['TimeStamp (sec)'].max()-df['TimeStamp (sec)'].min())
    except ValueError:
        totalseconds = 0

    try:
        totalmeters=int(df['cum_dist'].max())
    except ValueError:
        totalmeters = 0

    try:
        totalcalories=int(df[' Calories (kCal)'].max())
    except ValueError:
        totalcalories = 0

    try:
        avghr=int(df[' HRCur (bpm)'].mean())
    except ValueError:
        avghr = 1
    if avghr == 0:
        avghr=1
    try:
        maxhr=int(df[' HRCur (bpm)'].max())
    except ValueError:
        maxhr = 1
    if maxhr == 0:
        maxhr=1

    try:
        avgspm=int(df[' Cadence (stokes/min)'].mean())
    except ValueError:
        avgspm = 10

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

    timezero=arrow.get(tt).timestamp()
    if seconds[0]<timezero:
        dateobj=ps.parse(row_date)
        unixtimes=seconds+arrow.get(dateobj).timestamp() #time.mktime(dateobj.timetuple())

    top = Element('TrainingCenterDatabase')
    top.attrib['xmlns'] = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
    top.attrib['xmlns:ax'] = "http://www.garmin.com/xmlschemas/ActivityExtension/v2"
    top.attrib['xmlns:xsi'] = "http://www.w3.org/2001/XMLSchema-instance"
    top.attrib['xsi:schemaLocation'] = "http://www.garmin.com/xmlschemas/ActivityExtension/v2 http://www.garmin.com/xmlschemas/ActivityExtensionv2.xsd http://www.garmin.com/xmlschemas/FatCalories/v1 http://www.garmin.com/xmlschemas/fatcalorieextensionv1.xsd http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2 http://www.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd"

    activities = SubElement(top,'Activities')
    activity = SubElement(activities,'Activity')
    activity.attrib['Sport'] = sport
    id = SubElement(activity,'Id')
    id.text = row_date


    # Lap averages
    lap = SubElement(activity,'Lap')
    lap.attrib['StartTime'] = row_date
    totaltime = SubElement(lap,'TotalTimeSeconds')
    totaltime.text = '{s}'.format(s=totalseconds)
    distancemeters_el = SubElement(lap,'DistanceMeters')
    distancemeters_el.text = '{m}'.format(m=totalmeters)
    calories = SubElement(lap,'Calories')
    calories.text = '{c}'.format(c=totalcalories)
    avghr_el = SubElement(lap,'AverageHeartRateBpm')
    avghr_el.attrib['xsi:type'] = 'HeartRateInBeatsPerMinute_t'
    value = SubElement(avghr_el,'Value')
    value.text = '{s}'.format(s=avghr)

    maxhr_el = SubElement(lap,'MaximumHeartRateBpm')
    maxhr_el.attrib['xsi:type'] = 'HeartRateInBeatsPerMinute_t'
    value = SubElement(maxhr_el,'Value')
    value.text = '{s}'.format(s=maxhr)

    intensity = SubElement(lap,'Intensity')
    intensity.text = 'Active'

    cadence_el = SubElement(lap,'Cadence')
    cadence_el.text = '{s}'.format(s=avgspm)

    triggermethod = SubElement(lap,'TriggerMethod')
    triggermethod.text = 'Manual'

    track  = SubElement(lap,'Track')


    for i in range(nr_rows):
        hri=heartrate[i]
        if hri == 0:
            hri=1

        trackpoint = SubElement(track,'Trackpoint')

        s = arrow.get(unixtimes[i]).isoformat()
        time = SubElement(trackpoint,'Time')
        time.text = '{s}'.format(s=s)

        if (lat[i] != 0) and (lon[i] != 0):
            position = SubElement(trackpoint,'Position')
            latitudedegrees = SubElement(position,'LatitudeDegrees')
            latitudedegrees.text = '{lat}'.format(lat=lat[i])
            longitudedegrees = SubElement(position,'LongitudeDegrees')
            longitudedegrees.text = '{lon}'.format(lon=lon[i])

        dist = SubElement(trackpoint,'DistanceMeters')
        dist.text = '{d}'.format(d=distancemeters[i])

        hrbpm = SubElement(trackpoint,'HeartRateBpm')
        hrbpm.attrib['xsi:type'] = 'HeartRateInBeatsPerMinute_t'
        val = SubElement(hrbpm,'Value')
        val.text = '{s}'.format(s=hri)

        spm_el = SubElement(trackpoint,'Cadence')
        spm_el.text = '{s}'.format(s=cadence[i])

        if haspower:
            ext = SubElement(trackpoint,'Extensions')
            tpx = SubElement(ext,'TPX')
            tpx.attrib['xmlns'] = "http://www.garmin.com/xmlschemas/ActivityExtension/v2"
            watts = SubElement(tpx,'Watts')
            try:
                watts.text = '{s}'.format(s=int(power[i]))
            except ValueError:
                watts.text = 'NaN'


    notes = SubElement(activity,'Notes')
    notes.text = '{n}'.format(n=notes)

    creator = SubElement(top,'Creator')
    name = SubElement(creator,'Name')
    name.text = 'rowsandall.com/rowingdata'

    author = SubElement(top,'Author')
    author.attrib['xsi:type'] = 'Application_t'

    name = SubElement(author,'Name')
    name.text = 'rowingdata'

    build = SubElement(author,'Build')

    version = SubElement(build,'Version')

    versionmajor = SubElement(version,'VersionMajor')
    versionmajor.text = '0'

    versionminor = SubElement(version,'VersionMinor')
    versionminor.text = '75'

    type = SubElement(build,'Type')
    type.text = 'Release'

    lang = SubElement(author,'LangID')
    lang.text = 'EN'

    partnumber = SubElement(author,'PartNumber')
    partnumber.text = '000-00000-00'

    return prettify(top)

def write_tcx(tcxFile,df,row_date="2016-01-01",notes="Exported by rowingdata",
              sport="Other"):

    tcxtext = create_tcx(df,row_date=row_date, notes=notes, sport=sport)

    with open(tcxFile,'w') as fop:
        fop.write(tcxtext)


    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        xsd_file=six.moves.urllib.request.urlopen("https://www8.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd",context=ctx)
        output=open('TrainingCenterDatabasev2.xsd',textwritemode)
        if pythonversion <= 2:
            output.write(xsd_file.read().replace('\n',''))
        else:
            output.write(xsd_file.read().decode('utf-8').replace('\n',''))
        output.close()
        xsd_filename="TrainingCenterDatabasev2.xsd"

        # Run some tests
        try:
            tree=objectify.parse(tcxFile)
            try:
                schema=etree.XMLSchema(file=xsd_filename)
                parser=objectify.makeparser(schema=schema)
                objectify.fromstring(some_xml_string, parser)
                # print("YEAH!, your xml file has validated")
            except XMLSyntaxError:

                print("Oh NO!, your xml file does not validate")
                pass
        except:
            print("Oh NO!, your xml file does not validate")
            pass

    except six.moves.urllib.error.URLError:
        print("cannot download TCX schema")
        print("your TCX file is unvalidated. Good luck")




    return 1
