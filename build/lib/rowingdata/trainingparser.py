import numpy as np
import string
import re

def pieceparse(r):
    value = 0
    unit = 'seconds'
    p = re.compile('([0-9]*\.?[0-9]+)([A-Za-wyz]+)')
    m = p.match(r)
    if m != None:
	value = float(m.group(1))
	units = m.group(2)
	if units in ['meter','meters','m']:
	    unit = 'meters'
	if units in ['km','k']:
	    value*= 1000
	    unit = 'meters'
	if units in ['min','minute','minutes',"'"]:
	    unit = 'seconds'
	    value *= 60
	    
    # now check for 2:30min
    p = re.compile('([0-9]*)\:([0-9]+)([A-Za-wyz]+)')
    m = p.match(r)
    if m != None:
	unit = 'seconds'
	value1 = float(m.group(1))
	value2 = float(m.group(2))
	units = m.group(3)

	if units in ['h','hour','hrs']:
	    value = 3600*value1+60*value2
	else:
	    value = 60*value1+value2

    return [value,unit]

def parse(s,debug=0):
    if debug == 1:
	print s
	print ""

    s = s.strip()

    # first check for nx(aap noot)/5min
    p = re.compile('([0-9]+)x\(([A-Za-z]+)\)\/(.+)')
    m = p.match(s)
    if m != None:
	n = int(m.group(1))
	piece = m.group(2)
	rest = pieceparse(m.group(3))
	if n>1:
	    newstring = str(n-1)+"x("+piece+")/"+m.group(3)
	    piecestring = piece+"/"+m.group(3)
	    return parse(piecestring,debug=debug)+parse(newstring,debug=debug)
	if n==1:
	    return parse(piece+"/"+m.group(3),debug=debug)

    # now check for nxaap/5min
    p = re.compile('([0-9]+)x(.+)\/(.+)')
    m = p.match(s)
    if m != None:
	n = int(m.group(1))
	piece = m.group(2)
	rest = pieceparse(m.group(3))
	if n>1:
	    newstring = str(n-1)+"x"+piece+"/"+m.group(3)
	    piecestring = piece+"/"+m.group(3)
	    return parse(piecestring,debug=debug)+parse(newstring,debug=debug)
	if n==1:
	    return parse(piece+"/"+m.group(3),debug=debug)

    # now check for aap+noot
    p = re.compile('(.+)\+(.+)')
    m = p.match(s)
    if m != None:
	return parse(m.group(1),debug=debug)+parse(m.group(2),debug=debug)

    # now check for aap/noot
    p = re.compile('(.+)\/(.+)')
    m = p.match(s)
    if m != None:
	w = pieceparse(m.group(1))
	r = pieceparse(m.group(2))
	return [w[0],w[1],'work',r[0],r[1],'rest']

    # now check for 8x500m
    p = re.compile('([0-9]+)x(.+)')
    m = p.match(s)
    if m != None:
	n = int(m.group(1))
	if n>=2:
	    return parse(str(n-1)+"x"+m.group(2))+parse(m.group(2))
	if n == 1:
	    return parse(m.group(2))

    # now check for 2:30min
    p = re.compile('([0-9]*)\:([0-9]+)([A-Za-wyz]+)')
    m = p.match(s)
    if m != None:
	unit = 'seconds'
	value1 = float(m.group(1))
	value2 = float(m.group(2))
	units = m.group(3)

	if units in ['h','hour','hrs']:
	    value = 3600*value1+60*value2
	else:
	    value = 60*value1+value2

	return [value,unit,'work']

    # now check for 500m
    p = re.compile('([0-9]*\.?[0-9]+)([A-Za-wyz]+)')
    m = p.match(s)
    if m != None:
	unit = 'seconds'
	value = float(m.group(1))
	units = m.group(2)

	if units in ['meter','meters','m']:
	    unit = 'meters'
	if units in ['km','k']:
	    value*= 1000
	    unit = 'meters'
	if units in ['min','minute','minutes']:
	    unit = 'seconds'
	    value *= 60
	return [value,unit,'work']

    return [0,'seconds','rest']

def getlist(s,sel='value'):
    s1 = s[0:3]
    s2 = s[3:]

    if s2 != []:
	if sel == 'value':
	    return [s1[0]]+getlist(s2,sel=sel)
	if sel == 'unit':
	    return [s1[1]]+getlist(s2,sel=sel)
	if sel == 'type':
	    return [s1[2]]+getlist(s2,sel=sel)
    else:
	if sel == 'value':
	    return [s[0]]
	if sel == 'unit':
	    return [s[1]]
	if sel == 'type':
	    return [s[2]]

    return 0
    
