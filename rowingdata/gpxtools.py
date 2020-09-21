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

from lxml import etree
from docopt import docopt

ns1 = 'http://www.topografix.com/GPX/1/1'



def gpxtodf2(path):
    tree = etree.parse(path)
    root = tree.getroot()

    tracks = []
    lapnr = 0

    for element in root.iter():
        if element.tag == '{%s}trkseg'%ns1:
            lapnr += 1
        if element.tag == '{%s}trkpt'%ns1:
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
        latitude, longitude = element.values()
        lapid.append(lapnr)
        lat.append(float(latitude))
        lon.append(float(longitude))
        for child in element:
            if child.tag == '{%s}time'%ns1:
                try:
                    time = parser.parse(child.text)
                    timestamp = arrow.get(time).timestamp+arrow.get(time).microsecond/1.e6
                    t.append(timestamp)
                except KeyError:
                    t.append(np.nan)


    df = pd.DataFrame(
        {
            'TimeStamp (sec)':t,
            ' latitude':lat,
            ' longitude':lon,
            ' lapIdx':lapid,
            }
        )


    return df
