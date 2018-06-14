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

    outfile=readFile+"_o.csv"

    res=rowingdata.painsledDesktopParser(readFile)
    res.write_csv(outfile)

    row=rowingdata.rowingdata(outfile,rowtype="Indoor Rower",rower=rower)

    row.plotmeters_erg()

    print((row.allstats()))



    print(("done "+readFile))
