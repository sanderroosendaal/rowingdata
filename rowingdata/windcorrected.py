#! /usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
from . import rowingdata
from sys import argv
from six.moves import input

def main():
    readFile=argv[1]

    try:
        rowerFile=argv[2]
    except IndexError:
        rowerFile="defaultrower.txt"

    rower=rowingdata.getrower(rowerFile)

    try:
        boatFile=argv[3]
    except IndexError:
        boatFile="my1x.txt"


    csvoutput=readFile+'_p.csv'

    tcx=rowingdata.TCXParser(readFile)
    tcx.write_csv(csvoutput,window_size=20)

    res=rowingdata.rowingdata(csvoutput,rowtype="On-water",
                                rower=rower)

    s=input('Enter wind speed: ')
    windv=float(s)
    u=input('Enter wind speed units m=m/s, b=beaufort, k=knots: ')
    s=input('Enter wind bearing (N=0, E=90, S=180, W=270): ')
    winddirection=float(s)

    res.add_wind(windv,winddirection,units=u)
    res.add_bearing()

    res.otw_setpower(skiprows=9,rg=rowingdata.getrigging(boatFile))

    res.write_csv(csvoutput)
    
    res.plottime_otwpower()


    print(("done "+readFile))
