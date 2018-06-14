#! /usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
from . import rowingdata
from sys import argv

def main():
    readFile=argv[1]

    try:
        rowerFile=argv[2]
    except IndexError:
        rowerFile="defaultrower.txt"

    rower=rowingdata.getrower(rowerFile)

    csvoutput=readFile+"_o.CSV"

    rp=rowingdata.ErgDataParser(readFile)
    rp.write_csv(csvoutput)

    res=rowingdata.rowingdata(csvoutput,rowtype="On-water",
                                rower=rower)

    res.plottime_erg()

    print((res.allstats()))



    print(("done "+readFile))
