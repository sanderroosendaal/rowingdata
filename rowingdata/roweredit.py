#! /usr/bin/python
import rowingdata
from sys import argv

def main():
    try:
	rowerFile=argv[1]
    except IndexError:
	rowerFile="defaultrower.txt"

    print(rowingdata.roweredit(rowerFile))

    print("done")
