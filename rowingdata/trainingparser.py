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
    targetop = "@"


    lpar  = Literal( '(' ).suppress()
    rpar  = Literal( ')' ).suppress()
    num = Word(nums)
    num2 = Word(nums,exact=2)
    timeordist = Group(num+":"+num) | num
    ntimes = num+"x"
    unit = Word(alphas)
    interval = Group(timeordist+unit) | timeordist  # 5min
    target = Group(num+unit)

    multipleinterval = Group(ntimes+interval)  # 3x5min
    set = multipleinterval | interval  # 5min or 3x5min
    intervalwithrest = Group(set+"/"+interval) # 5min/3min or 3x5min/3min
    intervalwithtarget = Group(set+"@"+target)
    intervalwithtargetandrest = Group(intervalwithtarget+"/"+interval)
    expr = Forward()

    atom = intervalwithtargetandrest | intervalwithtarget | intervalwithrest | set | multipleinterval | interval | Group(lpar+expr+rpar)

    bigset = Group(ntimes+atom) | atom
    bigsetwithrest = Group(bigset+"/"+interval)
    bigsetwithtarget = Group(bigset+"@"+target)
    bigsetwithtargetandrest = Group(bigsetwithtarget+"/"+interval)
    majorset = bigsetwithtargetandrest | bigsetwithtarget | bigsetwithrest | bigset


    expr << majorset + ZeroOrMore( "+" + expr )
    return expr

def getintervalasdict(l,target=None):
    if len(l)==0:
        return [{}]
    elif len(l)==2:
        try:
            value = int(l[0])
            unit = l[1]
        except TypeError:
            valuemin = int(l[0][0])
            valuesec = int(l[0][2])
            value = 60*valuemin+valuesec
            unit = 'sec'
        d = {
            'value': value,
            'unit': unit,
            'type': 'work'
        }
        if target is not None:
            try:
                d['target'] = int(target[0])
                d['targetunit'] = target[1]
            except ValueError:
                pass
        return [d]
    elif len(l)==3 and l[1] == '/':
        a = getintervalasdict(l[0])
        b = getintervalasdict(l[2])
        b[0]['type'] = 'rest'
        return [a,b]
    elif len(l)==3 and l[1] == 'x':
        u = []
        u.append({'value':int(l[0]),'type':'repeatstart','unit':'m'})
        u.append(getintervalasdict(l[2]))
        u.append({'value':int(l[0]),'type':'repeat','unit':'m'})
        #for i in range(int(l[0])):
        #    u.append(getintervalasdict(l[2]))
        return u
    elif len(l)==3 and l[1] == '@':
        return getintervalasdict(l[0],target=l[2])
    elif len(l)==1:
        return getintervalasdict(l[0])
    else:
        return [getintervalasdict(l[0]),getintervalasdict(l[2:])]


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
    elif len(l)==3 and l[1] == '@':
        return getinterval(l[0])
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

def pieceparsedict(v):
    value = v['value']
    unit = 'seconds'
    if v['unit'] in ['meter','meters','m']:
        unit = 'meters'
    if v['unit'] in ['km','k','kilometer']:
        value *= 1000
        unit = 'meters'
    if v['unit'] in ['min','minute','minutes',"'"]:
        unit = 'seconds'
        value *= 60

    v['value'] = value
    v['unit'] = unit
    return v

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

def flattenlist(l):
    flatlist = []
    for sublist in l:
        if type(sublist)==dict:
            flatlist.append(pieceparsedict(sublist))
            continue
        elif type(sublist)==list:
            for item in flattenlist(sublist):
                flatlist.append(item)
        else:
            pass

    return(flatlist)

def parsetodict(s):
    try:
        r = Syntax().parseString(s).asList()
    except:
        return {}
    res = getintervalasdict(r)

    xres = flattenlist(res)

    return xres

def simpletofit(step,message_index=0,name=''):
    type = step['type']
    value = step['value']
    unit = step['unit']

    d = {
        'wkt_step_name': name,
        'stepId': message_index,
    }

    if unit == 'seconds':
        d['durationType'] = 'Time'
        d['durationValue'] = value*1000
    else:
        d['durationType'] = 'Distance'
        d['durationValue'] = value*100

    d['intensity'] = 'Active'
    if type == 'rest':
        d['intensity'] = 'Rest'

    try:
        target = step['target']
        targetunit = step['targetunit']
        if targetunit == 'W':
            d['targetType'] = 'Power'
            d['targetValue'] = target+1000
        if targetunit == 'spm':
            d['targetType'] = 'Cadence'
            d['targetValue'] = target
        if targetunit in ['bpm','hr']:
            d['targetType'] = 'HeartRate'
            d['targetValue'] = target+100
    except KeyError:
        pass

    return d

def repeatstep(step,startindex,name='',message_index=0):
    value = step['value']

    d = {
        'wkt_step_name': name,
        'stepId': message_index,
    }

    d['durationType'] = 'RepeatUntilStepsCmplt'
    d['targetValue'] = value
    d['durationValue'] = startindex

    return d

def tofitdict(steps,name='',sport='rowing'):
    newsteps = []
    message_index = 0
    repeatstack = []
    for step in steps:
        if step['type'] == 'repeatstart':
            repeatstack.append(message_index)
            continue
        if step['type'] == 'repeat':
            newsteps.append(repeatstep(step,repeatstack.pop(),message_index=message_index,name=str(message_index)))
            message_index = message_index+1
            continue
        newsteps.append(simpletofit(step,message_index=message_index,name=str(message_index)))
        message_index = message_index+1

    d = {
        'name':name,
        'sport':sport,
        'filename':'',
        'steps':newsteps
    }

    return d

def parse(s):
    try:
        r = Syntax().parseString(s).asList()
    except:
        return []

    res = getinterval(r)

    xres = []

    while len(res):
        xres += pieceparse(res[0:2]) + [res[2]]
        res = res[3:]

    return xres
