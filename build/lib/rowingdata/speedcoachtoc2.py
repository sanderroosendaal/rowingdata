#! /usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
from . import rowingdata
import time
from sys import argv

def main():
    readFile=argv[1]

    try:
	rowerFile=argv[2]
    except IndexError:
	rowerFile="defaultrower.txt"

    try:
	datestring=argv[3]
    except IndexError:
	datestring=time.strftime("%c")

    rower=rowingdata.getrower(rowerFile)
	

    res=rowingdata.speedcoachParser(readFile,row_date=datestring)

    file2=readFile+"_o.csv"

    res.write_csv(file2)

    row=rowingdata.rowingdata(file2,rowtype="On-water",rower=rower)

    row.uploadtoc2(rowerFile=rowerFile)



    print(("done "+readFile))
