#! /usr/bin/python
from __future__ import absolute_import
from . import rowingdata
from sys import argv


def main():
    readFile=argv[1]

    try:
        rowerFile=argv[2]
    except IndexError:
        rowerFile="defaultrower.txt"

    rower=rowingdata.getrower(rowerFile)

    row=rowingdata.rowingdata(readFile,rowtype="Indoor Rower",
                            rower=rower)

    rowingdata.copytocb(row.allstats())


