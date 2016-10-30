#!C:\Users\e408191\AppData\Local\Continuum\Anaconda\python.exe
import rowingdata
from sys import argv

readFile = argv[1]

try:
    rowerFile = argv[2]
except IndexError:
    rowerFile = "defaultrower.txt"

rower = rowingdata.getrower(rowerFile)

tcx = rowingdata.TCXParser(readFile)

file2 = readFile+"_o.csv"

tcx.write_csv(file2)

row = rowingdata.rowingdata(file2,rowtype="On-water",
			    rower=rower)

row.uploadtoc2(rowerFile=rowerFile)



print "done "+readFile
