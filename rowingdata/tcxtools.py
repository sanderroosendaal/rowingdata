import time
import iso8601
import numpy as np
import pandas as pd
from pandas import DataFrame
from lxml import objectify
import xmltodict as xd
from collections import OrderedDict as od
from dateutil import parser
import arrow


def tcx_getdict(path):
    with open(path,'r') as f:
        d = xd.parse(f)
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
    

def tcxtrack_getdata(track):
    trackpoints = tcxtrackgettrackpoint(track)
    df = pd.DataFrame(trackpoints)
    datetime = df['Time'].apply(lambda x:parser.parse(x,fuzzy=True))
    df['timestamp'] = datetime.apply(lambda x:arrow.get(x).timestamp+arrow.get(x).microsecond/1.e6)
    try:
        df['latitude'] = df['Position'].apply(lambda x:x['LatitudeDegrees'])
        df['longitude'] = df['Position'].apply(lambda x:x['LongitudeDegrees'])
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
            thekeys = extensionsdf.keys()
            try:
                l = extensionsdf['ax:ActivityTrackpointExtension']
                l = l.apply(pd.Series)['Extensions'].apply(pd.Series)
            except KeyError:
                l = {}
            for kk in l.keys():
                if kk != 0:
                    df[kk] = l[kk]
            for key in thekeys:
                l = extensionsdf[key].apply(pd.Series)
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
        for nr,lap in enumerate(laps):
            track = tcxlapgettrack(lap)
            dfi = tcxtrack_getdata(track)
            dfi['lapid'] = nr
            df = pd.concat([df,dfi])

    return df
