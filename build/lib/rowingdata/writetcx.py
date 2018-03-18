import time
import datetime
from dateutil import parser as ps
import lxml
import arrow
import numpy as np
from lxml import etree, objectify
from lxml.etree import XMLSyntaxError
import urllib2
import ssl

empty_tcx = """
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">
    <Activities>
        <Activity Sport="Other">
            <Id>2015-03-28T20:45:15.000Z</Id>
            <Creator xsi:type="Device_t" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <Name>Empty File</Name>
                <UnitId>0</UnitId>
                <ProductID>0</ProductID>
            </Creator>
        </Activity>
    </Activities>
</TrainingCenterDatabase>   
"""

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


def write_tcx(tcxFile,df,row_date="2016-01-01",notes="Exported by rowingdata"):
    if notes==None:
        notes="Exported by rowingdata"
    notes = notes.encode('utf-8')
    f=open(tcxFile,'w')
    
    totalseconds=int(df['TimeStamp (sec)'].max()-df['TimeStamp (sec)'].min())
    totalmeters=int(df['cum_dist'].max())
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
	long=df[' longitude'].values
    except KeyError:
	long=np.zeros(nr_rows)

    haspower=1

    try:
	power=df[' Power (watts)'].values
    except KeyError:
	haspower=0
	
    s="2000-01-01"
    tt=ps.parse(s)
    #timezero=time.mktime(tt.timetuple())
    timezero=arrow.get(tt).timestamp
    if seconds[0]<timezero:
	# print("Taking Row_Date ",row_date)
	dateobj=ps.parse(row_date)
	unixtimes=seconds+arrow.get(dateobj).timestamp #time.mktime(dateobj.timetuple())


    
    f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n')
    f.write('<TrainingCenterDatabase')
    f.write('  xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"\n')
    f.write('  xmlns:ax="http://www.garmin.com/xmlschemas/ActivityExtension/v2"\n')
    f.write('  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
    f.write('xsi:schemaLocation="http://www.garmin.com/xmlschemas/ActivityExtension/v2 http://www.garmin.com/xmlschemas/ActivityExtensionv2.xsd http://www.garmin.com/xmlschemas/FatCalories/v1 http://www.garmin.com/xmlschemas/fatcalorieextensionv1.xsd http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2 http://www.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd">\n')
    f.write('  <Activities>\n')
    f.write('    <Activity Sport="Other">\n')

    datetimestring=row_date

    f.write('      <Id>{s}</Id>\n'.format(s=datetimestring))

    lap_begin(f,datetimestring,totalmeters,avghr,maxhr,avgspm,totalseconds)

    for i in range(nr_rows):
	hri=heartrate[i]
	if hri == 0:
	    hri=1
	f.write('          <Trackpoint>\n')
	#s=datetime.datetime.fromtimestamp(unixtimes[i]).isoformat()
        s = arrow.get(unixtimes[i]).isoformat()
	f.write('            <Time>{s}</Time>\n'.format(s=s))
	if (lat[i] != 0) & (long[i] != 0 ):
	    f.write('            <Position>\n')
	    f.write('              <LatitudeDegrees>{lat}</LatitudeDegrees>\n'.format(
		lat=lat[i]
		))
	    f.write('              <LongitudeDegrees>{long}</LongitudeDegrees>\n'.format(
		long=long[i]
		))
	    f.write('            </Position>\n')
	f.write('            <DistanceMeters>{d}</DistanceMeters>\n'.format(
	    d=distancemeters[i]
	    ))
	f.write('            <HeartRateBpm xsi:type="HeartRateInBeatsPerMinute_t">\n')
	f.write('              <Value>{h}</Value>\n'.format(h=hri))
	f.write('            </HeartRateBpm>\n')
	f.write('            <Cadence>{c}</Cadence>\n'.format(c=cadence[i]))
	if haspower:
	    f.write('            <Extensions>\n')
	    f.write('              <TPX xmlns="http://www.garmin.com/xmlschemas/ActivityExtension/v2">\n')
	    f.write('                <Watts>{p}</Watts>\n'.format(p=int(power[i])))
	    f.write('              </TPX>\n')
	    f.write('            </Extensions>\n')
	f.write('          </Trackpoint>\n')



    f.write('          </Track>\n')
    f.write('        </Lap>\n')
    f.write('      <Notes>'+notes+'</Notes>\n')
    f.write('    </Activity>\n')
    f.write('  </Activities>\n')
    f.write('  <Creator>\n')
    f.write('    <Name>rowsandall.com/rowingdata</Name>\n')
    f.write('  </Creator>\n')
    f.write('  <Author xsi:type="Application_t">\n')
    f.write('    <Name>rowingdata</Name>\n')
    f.write('    <Build>\n')
    f.write('      <Version>\n')
    f.write('        <VersionMajor>0</VersionMajor>\n')
    f.write('        <VersionMinor>75</VersionMinor>\n')
    f.write('      </Version>\n')
    f.write('      <Type>Release</Type>\n')
    f.write('    </Build>\n')
    f.write('    <LangID>EN</LangID>\n')
    f.write('    <PartNumber>000-00000-00</PartNumber>\n')
    f.write('  </Author>\n')
    f.write('</TrainingCenterDatabase>\n')

    f.close()

    file=open(tcxFile,'r')
    
    some_xml_string=file.read()

    file.close()
    
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
	xsd_file=urllib2.urlopen("https://www8.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd",context=ctx)
	output=open('TrainingCenterDatabasev2.xsd','w') 
	output.write(xsd_file.read().replace('\n',''))
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
            print("Oh NO!, your xmsl file does not validate")
            pass
	
    except urllib2.URLError:
	print("cannot download TCX schema")
	print("your TCX file is unvalidated. Good luck")

    
	    

    return 1

