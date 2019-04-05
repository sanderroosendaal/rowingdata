from __future__ import absolute_import
import numpy as np
import string
import re
from six.moves import range


def cleanzeros(values):
    newlist = []
    oldlist = values
    while oldlist:
        interval = oldlist[0:3]
        if interval[0] != 0:
            newlist += interval
        oldlist = oldlist[3:]

    return newlist

from pyparsing import (
    Literal,Word,ZeroOrMore,Forward,nums,oneOf,Group,
    alphas
    )

def Syntax():
    op = "+"
    restop = "/"
    times = "x"
    minsep = ":"


    lpar  = Literal( '(' ).suppress()
    rpar  = Literal( ')' ).suppress()
    num = Word(nums)
    num2 = Word(nums,exact=2)
    timeordist = Group(num2+":"+num2) | num
    ntimes = num+"x"
    unit = Word(alphas)
    interval = Group(timeordist+unit) | timeordist  # 5min
    
    multipleinterval = Group(ntimes+interval)  # 3x5min
    set = multipleinterval | interval  # 5min or 3x5min
    intervalwithrest = Group(set+"/"+interval) # 5min/3min or 3x5min/3min
    expr = Forward()

    atom = intervalwithrest | set | multipleinterval | interval | Group(lpar+expr+rpar)

    bigset = Group(ntimes+atom) | atom
    bigsetwithrest = Group(bigset+"/"+interval)
    majorset = bigsetwithrest | bigset

    
    expr << majorset + ZeroOrMore( "+" + expr )
    return expr


def getinterval(l):
    if len(l)==0:
        return []
    elif len(l)==2:
        try:
            value = int(l[0])
            unit = l[1]
        except TypeError:
            valuemin = int(l[0][0])
            valuesec = int(l[0][2])
            value = 60*valuemin+valuesec
            unit = 'sec'
        return [value,unit,'work']
    elif len(l)==3 and l[1] == '/':
        a = getinterval(l[0])
        b = getinterval(l[2])
        b[2] = 'rest'
        return a+b
    elif len(l)==3 and l[1] == 'x':
        u = []
        for i in range(int(l[0])):
            u+=getinterval(l[2])
        return u
    elif len(l)==1:
        return getinterval(l[0])
    else:
        return getinterval(l[0])+getinterval(l[2:])

def pieceparse(v):
    value = int(v[0])
    unit = 'seconds'
    if v[1] in ['meter','meters','m']:
        unit = 'meters'
    if v[1] in ['km','k','kilometer']:
        value *= 1000
        unit = 'meters'
    if v[1] in ['min','minute','minutes',"'"]:
        unit = 'seconds'
        value *= 60

    return [value,unit]

def getlist(s,sel='value'):
    s1=s[0:3]
    s2=s[3:]

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

def parse(s):
    r = Syntax().parseString(s)
    if len(r)==2:
        res =  getinterval(r)
    elif len(r)==1:
        res =  getinterval(r)
    else:
        res =  getinterval(r[0])+getinterval(r[2:])

    xres = []

    while len(res):
        xres += pieceparse(res[0:2]) + [res[2]]
        res = res[3:]

    return xres
