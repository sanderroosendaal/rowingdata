#! /usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
from . import rowingdata
from sys import argv

def main():
    try:
	boatFile=argv[1]
    except IndexError:
	boatFile="my1x.txt"

    print((rowingdata.boatedit(boatFile)))

    print("done")
