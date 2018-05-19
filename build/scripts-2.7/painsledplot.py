#!C:\Users\e408191\AppData\Local\Continuum\Anaconda\python.exe
from __future__ import absolute_import
from __future__ import print_function
import rowingdata
from sys import argv


readFile = argv[1]

try:
    rowerFile = argv[2]
except IndexError:
    rowerFile = "defaultrower.txt"

rower = rowingdata.getrower(rowerFile)

row = rowingdata.rowingdata(readFile,rowtype="Indoor Rower",
			    rower=rower)

print(row.allstats())

row.plotmeters_erg()




print("done "+readFile)
