#!C:\Users\e408191\AppData\Local\Continuum\Anaconda\python.exe
import rowingdata
from sys import argv

readFile = argv[1]

try:
    rowerFile = argv[2]
except IndexError:
    rowerFile = "defaultrower.txt"

rower = rowingdata.getrower(rowerFile)

row = rowingdata.rowingdata(readFile,rower=rower)

row.uploadtoc2(rowerFile=rowerFile)



print "done "+readFile
