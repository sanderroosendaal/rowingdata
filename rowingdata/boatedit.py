#! /usr/bin/python
import rowingdata
from sys import argv

def main():
    try:
	boatFile=argv[1]
    except IndexError:
	boatFile="my1x.txt"

    print(rowingdata.boatedit(boatFile))

    print("done")
