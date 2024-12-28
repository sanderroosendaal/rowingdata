# pylint: disable=C0103
from __future__ import absolute_import
import pandas as pd
import re
import sys
import unicodedata
import codecs
import xmltodict as xd
from dateutil import parser
import arrow
import gzip
import numpy as np
import string
from six import unichr
from datetime import datetime, timedelta

from lxml import etree
from lxml.etree import XMLSyntaxError
from docopt import docopt
import xml.etree.ElementTree as ET

ns1 = 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'
ns2 = 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'

import unicodedata
def clean_string(input):
    if input:
        RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
            u'|' + \
            u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
            (unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
             unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
             unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
             )
        
        input = re.sub(RE_XML_ILLEGAL, "", input)
        input = input.replace(RE_XML_ILLEGAL,'').strip()
    return input

def clean_xml_content(xml_content):
    # Clean the XML content of potentially problematic characters
    return ''.join(char for char in xml_content if char.isprintable())

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def strip_control_characters(input):
    if input:

        # unicode invalid characters
        RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                         u'|' + \
                         u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                          (unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                           unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                           unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                           )

        input = re.sub(RE_XML_ILLEGAL, "", input)



        #input = "".join(ch for ch in input if unicodedata.category(ch)[0]!="C")


        # ascii control characters
        #input = re.sub(r"[\x01-\x1F\x7F]", "", input)

    return input

def tcx_getdict(path):
    extension = path[-3:].lower()
    if extension == '.gz':
        with gzip.open(path,'r') as f:
            input = f.read()
            input = strip_control_characters(input)
            d = xd.parse(input)
    else:
        with open(path, 'r') as f:
            input = f.read()
            input = strip_control_characters(input)
            d = xd.parse(input)
    return d['TrainingCenterDatabase']

def tcxgetactivities(d):
    try:
        return d['Activities']['Activity']
    except KeyError:
        return None

def tcxactivitygetid(d):
    try:
        return d['Id']
    except KeyError:
        return None

def tcxactivitygetlaps(d):
    try:
        return d['Lap']
    except KeyError:
        return None
    except TypeError:
        try:
            return d[0]['Lap']
        except KeyError:
            return None

def tcxlapgettrack(d):
    try:
        return d['Track']
    except KeyError:
        return None

def tcxtrackgettrackpoint(d):
    try:
        return d['Trackpoint']
    except KeyError:
        return None
    except TypeError:
        try:
            return d[0]['Trackpoint']
        except KeyError:
            return None

def getvalue(x,key):
    try:
        return x[key]
    except TypeError:
        return np.nan

def tcxtrack_getdata(track):
    trackpoints = tcxtrackgettrackpoint(track)
    df = pd.DataFrame(trackpoints)
    datetime = df['Time'].apply(lambda x: parser.parse(x, fuzzy=True))
    df['timestamp'] = datetime.apply(
        lambda x: arrow.get(x).timestamp()+arrow.get(x).microsecond/1.e6
    )
    try:
        #df['latitude'] = df['Position'].apply(lambda x: x['LatitudeDegrees'])
        #df['longitude'] = df['Position'].apply(lambda x: x['LongitudeDegrees'])
        df['latitude'] = df['Position'].apply(
            lambda x: getvalue(x,'LatitudeDegrees'))
        df['longitude'] = df['Position'].apply(
            lambda x: getvalue(x,'LongitudeDegrees'))
    except KeyError:
        pass
    except TypeError:
        pass

    for key in df.keys():
        v = df[key].dropna().values
        try:
            if len(v) and 'Value' in v[0]:
                l = df[key].apply(pd.Series)
                df[key] = l['Value']
        except TypeError:
            pass

        if key == 'Extensions':
            extensionsdf = df[key].apply(pd.Series)
            thekeys = list(extensionsdf.keys())
            for counter, key in enumerate(thekeys):
                if key:
                    df['extension'+str(counter)] = key
                    l = extensionsdf[key].apply(pd.Series)
                    if 'Extensions' in list(l.keys()):
                        #print 'aap'
                        l = l.apply(pd.Series)['Extensions'].apply(pd.Series)
                    for kk in l.keys():
                        if kk != 0 and 'xmlns' not in kk:
                            df[kk] = l[kk]

    return df

def tcxtodf(path):

    data = tcx_getdict(path)
    activity = tcxgetactivities(data)

    laps = tcxactivitygetlaps(activity)

    try:
        track = tcxlapgettrack(laps)
        df = tcxtrack_getdata(track)
    except TypeError:
        df = pd.DataFrame()
        for nr, lap in enumerate(laps):
            track = tcxlapgettrack(lap)
            dfi = tcxtrack_getdata(track)
            dfi['lapid'] = nr
            df = pd.concat([df, dfi])

    return df

def process_trackpoint(trackpoint):
    trackp = {}
    for child in trackpoint:
        for elem in child.iter():
            if elem.tag == '{%s}Time'%ns1:
                trackp['time'] = elem.text
            if elem.tag == '{%s}DistanceMeters'%ns1:
                    trackp['distance'] = float(elem.text)
            if elem.tag == '{%s}Cadence'%ns1:
                try:
                    trackp['cadence'] = float(elem.text)
                except TypeError:
                    trackp['cadence'] = 0.
            if elem.tag == '{%s}HeartRateBpm'%ns1:
                for hrchild in elem:
                    if hrchild.tag == '{%s}Value'%ns1:
                        try:
                            trackp['hr'] = int(hrchild.text)
                        except TypeError:
                            trackp['hr'] = 0
            if elem.tag == '{%s}Extensions'%ns1:
                for extchild in elem:
                    if extchild.tag == '{%s}TPX'%ns2:
                        for pchild in extchild:
                            if pchild.tag == '{%s}Watts'%ns2:
                                try:
                                    trackp['power'] = float(pchild.text)
                                except TypeError:
                                    trackp['power'] = 0
            if elem.tag == '{%s}Position'%ns1:
                for poschild in elem:
                    if poschild.tag == '{%s}LatitudeDegrees'%ns1:
                        trackp['latitude'] = float(poschild.text)
                    if poschild.tag == '{%s}LongitudeDegrees'%ns1:
                        trackp['longitude'] = float(poschild.text)


    return trackp

import os

def tcxtodf2(path):
    extension = path[-3:].lower()
    #p = etree.XMLParser(recover=True)
    p = etree.XMLParser()
    try:
        if extension == '.gz':
            with gzip.open(path,'r') as f:
                input = f.read()
                input = input.lstrip()
                input = strip_control_characters(input)
        else:
            with open(path, 'r') as f:
                input = f.read()
                input = input.lstrip()
                input = strip_control_characters(input)

            with open('temp_xml.tcx','w') as f:
                f.write(input)

            tree = etree.parse('temp_xml.tcx',parser=p)
            os.remove('temp_xml.tcx')
    except (TypeError,XMLSyntaxError):
        tree = etree.parse(path,parser=p)

    root = tree.getroot()

    tracks = []
    lapnr = 0

    if root is None:
        return pd.DataFrame()
    for element in root.iter():
        if element.tag == '{%s}Lap'%ns1:
            lapnr += 1
        if element.tag == '{%s}Track'%ns1:
            tracks.append((lapnr,element))

    t = []
    d = []
    hr = []
    power = []
    lat = []
    cadence = []
    lon = []
    lapid = []

    for lapnr,element in tracks:
        for child in element:
            if child.tag == '{%s}Trackpoint'%ns1:
                trackp = process_trackpoint(child)
                try:
                    time = parser.parse(trackp['time'])
                    timestamp = arrow.get(time).timestamp()+arrow.get(time).microsecond/1.e6
                    t.append(timestamp)
                except KeyError:
                    t.append(np.nan)

                try:
                    d.append(trackp['distance'])
                except KeyError:
                    d.append(np.nan)

                try:
                    cadence.append(trackp['cadence'])
                except KeyError:
                    cadence.append(np.nan)

                try:
                    hr.append(trackp['hr'])
                except KeyError:
                    hr.append(0)

                try:
                    power.append(trackp['power'])
                except KeyError:
                    power.append(0)

                try:
                    lat.append(trackp['latitude'])
                except KeyError:
                    lat.append(np.nan)

                try:
                    lon.append(trackp['longitude'])
                except KeyError:
                    lon.append(np.nan)

                lapid.append(lapnr)


    df = pd.DataFrame(
        {
            'timestamp':t,
            'HeartRateBpm':hr,
            'DistanceMeters':d,
            'Cadence':cadence,
            'Watts':power,
            'latitude':lat,
            'longitude':lon,
            'lapid':lapid,
            }
        )

    df['Speed'] = df['DistanceMeters'].diff()/df['timestamp'].diff()
    df.loc[0,'Speed'] = 0

    return df

def tcxtodf3(path):
    try:
        if path.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as gz_file:
                # Parse XML data from the gzipped TCX file
                xml_content = gz_file.read()
                cleaned_xml_content = clean_string(xml_content)
                tree = ET.fromstring(cleaned_xml_content)
        else:
            # Parse XML data from the regular TCX file
            with open(path, 'r', encoding='utf-8', errors='replace') as tcx_file:
                xml_content = tcx_file.read()
                cleaned_xml_content = clean_string(xml_content)
                tree = ET.fromstring(cleaned_xml_content)
            
        
        activity_data = []
        lap_id = 0
        for lap_node in tree.findall(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Lap"):
            lap_id += 1
            intensity_node = lap_node.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Intensity")
            workoutstate = 4
            intensity = clean_string(intensity_node.text) if intensity_node is not None else ""
            if intensity == "Actve":
                workoutstate = 4
            elif intensity == "Resting":
                workoutstate = 3
            for trackpoint in lap_node.findall(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Trackpoint"):
                time_node = trackpoint.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Time")
                timestamp_str = clean_string(time_node.text)
                time = parser.parse(timestamp_str)
                timestamp = arrow.get(time).timestamp()+arrow.get(time).microsecond/1.e6
                #timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")

                watts = 0
                check_factor = 0
                extensions_element = trackpoint.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Extensions")
                if extensions_element is not None:
                    # print child elements of extensions_element
                    watts_node = extensions_element.find(".//{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX/{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Watts")
                    
                    watts = float(clean_string(watts_node.text)) if watts_node is not None else 0

                    check_factor = extensions_element.find(".//{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX/{http://www.garmin.com/xmlschemas/ActivityExtension/v2}Check")
                    try:
                        check_factor = float(clean_string(check_factor.text)) if check_factor is not None else 0
                    except ValueError:
                        check_factor = 0

                speed_node = trackpoint.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Speed")
                speed = float(clean_string(speed_node.text)) if speed_node is not None else 0

                    

                cadence_node = trackpoint.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Cadence")
                try:
                    cadence = int(clean_string(cadence_node.text)) if cadence_node is not None else 0
                except ValueError:
                    cadence = float(clean_string(cadence_node.text)) if cadence_node is not None else 0


                hr_node = trackpoint.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}HeartRateBpm/{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Value")
                try:
                    heart_rate = int(clean_string(hr_node.text)) if hr_node is not None else 0
                except ValueError:
                    heart_rate = float(clean_string(hr_node.text)) if hr_node is not None else 0

                distance_node = trackpoint.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters")
                distance = float(clean_string(distance_node.text)) if distance_node is not None else 0
                
                latitude_node = trackpoint.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Position/{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LatitudeDegrees")
                longitude_node = trackpoint.find(".//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Position/{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}LongitudeDegrees")
                latitude = float(clean_string(latitude_node.text)) if latitude_node is not None else 0
                longitude = float(clean_string(longitude_node.text)) if longitude_node is not None else 0
 
                activity_data.append({
                    'timestamp': timestamp,
                    'latitude': latitude,
                    'longitude': longitude,
                    'HeartRateBpm': heart_rate,
                    'DistanceMeters': distance,
                    'Cadence': cadence,
                    'Watts': watts, 
                    'lapid': lap_id,
                    'Speed': speed,
                    'check_factor': check_factor,
                    ' WorkoutState': workoutstate
                })

        # Create a Pandas DataFrame from the activity data
        df = pd.DataFrame(activity_data)
        return df
    except ET.ParseError as e:
        return pd.DataFrame()
