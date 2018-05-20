#! /usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
from . import rowingdata
from sys import argv

def main():
    try:
        rowerFile=argv[1]
    except IndexError:
        rowerFile="defaultrower.txt"

    print((rowingdata.roweredit(rowerFile)))

    print("done")
