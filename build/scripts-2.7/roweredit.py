#!C:\Users\e408191\AppData\Local\Continuum\Anaconda\python.exe
from __future__ import absolute_import
from __future__ import print_function
import rowingdata
from sys import argv


try:
    rowerFile = argv[1]
except IndexError:
    rowerFile = "defaultrower.txt"

print(rowingdata.roweredit(rowerFile))

print("done")
