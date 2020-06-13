from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import grid
from matplotlib import figure
from matplotlib.ticker import MultipleLocator,FuncFormatter,NullFormatter

from . import rowingphysics
from six.moves import map
from six.moves import range

def format_time_tick(x,pos=None):
	hour=int(x/3600)
	min=int((x-hour*3600.)/60)
	min_str=str(min).zfill(2)
	template='%d:%s'
	return template % (hour,min_str)

def format_time(x,pos=None):


    min = int(x/60.)
    sec = x-min*60.

    str1 = "{min:0>2}:{sec:0>4.1f}".format(
	min=min,
	sec=sec,
	)

    return str1

def format_pace_tick(x,pos=None):
	min=int(x/60.)
	sec=int(x-min*60.)
	sec_str=str(sec).zfill(2)
	template='%d:%s'
	return template % (min,sec_str)

def format_pace(x,pos=None):
    if np.isinf(x) or np.isnan(x):
	x=0
	
    min=int(x/60)
    sec=(x-min*60.)

    str1 = "{min:0>2}:{sec:0>4.1f}".format(
	min = min,
	sec = sec
	)

    return str1

def format_time(x,pos=None):


    min = int(x/60.)
    sec = x-min*60

    str1 = "{min:0>2}:{sec:0>4.1f}".format(
	min=min,
	sec=sec,
	)

    return str1

def tempofromergsplit(ergsplit):
    tempo1 = 25.
    tempo2 = 35.

    split1 = 120.
    split2 = 85.

    ratio = (ergsplit-split2)/(split1-split2)

    tempo = tempo2+ratio*(tempo1-tempo2)

    return tempo

def tempofromotwsplit(otwsplit,otwsplits,mc=70.):
    tempo1 = 16.
    tempo2 = 35.

    split1 = max(rawtoseconds(otwsplits))/4.
    split2 = min(rawtoseconds(otwsplits))/4.

    ratio = (otwsplit-split2)/(split1-split2)

    tempo = tempo2+ratio*(tempo1-tempo2)

#    if tempo<18:
#	tempo = 18.0

    return tempo

def splitvalues(s):
    min, sec = list(map(float, s.split(':')))

    return [min,sec]

def rawtoseconds(raw):
    r = np.array(list(map(splitvalues,raw)))
    thetime = 60.*r[:,0]+r[:,1]
    return thetime

def plotdata(filename,r,rg,erg):
    expdata_raw = pd.read_csv(filename)

    ergtime = rawtoseconds(expdata_raw['erg score'])/4.

    fifty = rawtoseconds(expdata_raw['50kg'])/4.
    sixty = rawtoseconds(expdata_raw['60kg'])/4.
    seventy = rawtoseconds(expdata_raw['70kg'])/4.
    eighty = rawtoseconds(expdata_raw['80kg'])/4.
    ninety = rawtoseconds(expdata_raw['90kg'])/4.
    hundred = rawtoseconds(expdata_raw['100kg'])/4.
    hundredten = rawtoseconds(expdata_raw['110kg'])/4.

    ratio = 0.5

    print('50kg')

    r.mc = 50.0
    r.tempo = 30.

    otwsplit50kg = []

    for ergscore in expdata_raw['erg score']:
	print(ergscore)
	mins, secs = splitvalues(ergscore)
	ergsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromergsplit(ergsplit)
	ergsplitmin = int(ergsplit) / 60
	ergsplitsec = ergsplit-60*ergsplitmin

	res = rowingphysics.ergtopower(ergsplitmin,ergsplitsec,ratio,r,erg)
	totalpower = res[0]
	ergpower = res[1]
	res = rowingphysics.constantwatt(totalpower,r,rg)
	otwsplit = 500./res[1]
	otwsplit50kg.append(otwsplit)

    print('70kg')

    r.mc = 70.0

    otwsplit70kg = []

    for ergscore in expdata_raw['erg score']:
	print(ergscore)
	mins, secs = splitvalues(ergscore)
	ergsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromergsplit(ergsplit)
	ergsplitmin = int(ergsplit) / 60
	ergsplitsec = ergsplit-60*ergsplitmin

	res = rowingphysics.ergtopower(ergsplitmin,ergsplitsec,ratio,r,erg)
	totalpower = res[0]
	ergpower = res[1]
	res = rowingphysics.constantwatt(totalpower,r,rg)
	otwsplit = 500./res[1]
	otwsplit70kg.append(otwsplit)

    print('90kg')

    r.mc = 90.0

    otwsplit90kg = []

    for ergscore in expdata_raw['erg score']:
	print(ergscore)
	mins, secs = splitvalues(ergscore)
	ergsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromergsplit(ergsplit)
	ergsplitmin = int(ergsplit) / 60
	ergsplitsec = ergsplit-60*ergsplitmin

	res = rowingphysics.ergtopower(ergsplitmin,ergsplitsec,ratio,r,erg)
	totalpower = res[0]
	ergpower = res[1]
	res = rowingphysics.constantwatt(totalpower,r,rg)
	otwsplit = 500./res[1]
	otwsplit90kg.append(otwsplit)

    print('110kg')

    r.mc = 110.0

    otwsplit110kg = []

    for ergscore in expdata_raw['erg score']:
	print(ergscore)
	mins, secs = splitvalues(ergscore)
	ergsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromergsplit(ergsplit)
	ergsplitmin = int(ergsplit) / 60
	ergsplitsec = ergsplit-60*ergsplitmin

	res = rowingphysics.ergtopower(ergsplitmin,ergsplitsec,ratio,r,erg)
	totalpower = res[0]
	ergpower = res[1]
	res = rowingphysics.constantwatt(totalpower,r,rg)
	otwsplit = 500./res[1]
	otwsplit110kg.append(otwsplit)

    # making the plot

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1)
    ax.plot(ergtime,fifty,color='k')
    ax.plot(ergtime,otwsplit50kg,color='k')
#    ax.plot(ergtime,sixty,color='r')
    ax.plot(ergtime,seventy,color='g')
    ax.plot(ergtime,otwsplit70kg,color='g')
#    ax.plot(ergtime,eighty,color='b')
    ax.plot(ergtime,ninety,color='c')
    ax.plot(ergtime,otwsplit90kg,color='c')
#    ax.plot(ergtime,hundred,color='m')
    ax.plot(ergtime,hundredten,color='y')
    ax.plot(ergtime,otwsplit110kg,color='y')

    ax.axis([125,75,150,75])
    ax.set_xticks(list(range(85,125,10)))
    ax.set_xlabel('Erg split')
    ax.set_ylabel('OTW split')
    ax.set_title(filename)
    ax.set_yticks(list(range(85,150,10)))
    timeTickFormatter = NullFormatter()

    majorTimeFormatter = FuncFormatter(format_time_tick)
    majorLocator = (15*60)
    ax.xaxis.set_major_formatter(majorTimeFormatter)

    majorFormatter = FuncFormatter(format_pace_tick)
    majorLocator = (5)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_major_formatter(majorFormatter)

    ax.legend(['50kg','50kg  OTW','70kg','70kg OTW',
	       '90kg','90kg OTW',
	       '110kg','110kg OTW'],
	      prop={'size':10},loc=0)

    grid(True)

    fig.show()

def plotinversedata(filename,r,rg,erg):
    expdata_raw = pd.read_csv(filename)

    ergtime = rawtoseconds(expdata_raw['erg score'])/4.

    fifty = rawtoseconds(expdata_raw['50kg'])/4.
    sixty = rawtoseconds(expdata_raw['60kg'])/4.
    seventy = rawtoseconds(expdata_raw['70kg'])/4.
    eighty = rawtoseconds(expdata_raw['80kg'])/4.
    ninety = rawtoseconds(expdata_raw['90kg'])/4.
    hundred = rawtoseconds(expdata_raw['100kg'])/4.
    hundredten = rawtoseconds(expdata_raw['110kg'])/4.

    ratio = 0.5

    print('50kg')

    r.mc = 50.0
    r.tempo = 30.

    ergsplit50kg = []

    for otwscore in expdata_raw['50kg']:
	print(otwscore)
	mins, secs = splitvalues(otwscore)
	otwsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromotwsplit(otwsplit,expdata_raw['50kg'],50)
	otwsplitmin = int(otwsplit) / 60
	otwsplitsec = otwsplit-60*otwsplitmin

	velo = 500./float(otwsplit)

	res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
	force = res[0]
	power = res[3]
	ratio = res[2]
	try:
	    reserg = rowingphysics.constantwatt_erg(power,r,erg,theconst=1.0,
						    aantal=20,aantal2=20,
						    ratiomin=0.4,ratiomax=0.6)
	except:
	    ergpower = 50.
	    reserg = [0,1,2,3,50]
	    
	ergpower = reserg[4]
	ergvelo = (ergpower/2.8)**(1./3.)
	ergpace = 500./ergvelo
	print((format_pace(otwsplit),format_pace(ergpace)))
	ergsplit50kg.append(ergpace)


    print('70kg')

    r.mc = 70.0
    r.tempo = 30.

    ergsplit70kg = []

    for otwscore in expdata_raw['70kg']:
	print(otwscore)
	mins, secs = splitvalues(otwscore)
	otwsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromotwsplit(otwsplit,expdata_raw['70kg'],70)
	otwsplitmin = int(otwsplit) / 60
	otwsplitsec = otwsplit-60*otwsplitmin

	velo = 500./float(otwsplit)

	res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
	force = res[0]
	power = res[3]
	ratio = res[2]
	try:
	    reserg = rowingphysics.constantwatt_erg(power,r,erg,theconst=1.0,
						    aantal=20,aantal2=20,
						    ratiomin=0.4,ratiomax=0.6)

	except:
	    ergpower = 50.

	ergpower = reserg[4]
	ergvelo = (ergpower/2.8)**(1./3.)
	ergpace = 500./ergvelo
	print((format_pace(otwsplit),format_pace(ergpace)))
	ergsplit70kg.append(ergpace)


    print('90kg')

    r.mc = 90.0
    r.tempo = 30.

    ergsplit90kg = []

    for otwscore in expdata_raw['90kg']:
	print(otwscore)
	mins, secs = splitvalues(otwscore)
	otwsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromotwsplit(otwsplit,expdata_raw['90kg'],90)
	otwsplitmin = int(otwsplit) / 60
	otwsplitsec = otwsplit-60*otwsplitmin

	velo = 500./float(otwsplit)

	res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
	force = res[0]
	power = res[3]
	ratio = res[2]
	try:
	    reserg = rowingphysics.constantwatt_erg(power,r,erg,theconst=1.0,
						    aantal=20,aantal2=20,
						    ratiomin=0.4,ratiomax=0.6)
	except:
	    ergpower = 50.

	ergpower = reserg[4]
	ergvelo = (ergpower/2.8)**(1./3.)
	ergpace = 500./ergvelo
	print((format_pace(otwsplit),format_pace(ergpace)))
	ergsplit90kg.append(ergpace)


    print('110kg')

    r.mc = 110.0
    r.tempo = 30.

    ergsplit110kg = []

    for otwscore in expdata_raw['110kg']:
	print(otwscore)
	mins, secs = splitvalues(otwscore)
	otwsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromotwsplit(otwsplit,expdata_raw['110kg'],110)
	otwsplitmin = int(otwsplit) / 60
	otwsplitsec = otwsplit-60*otwsplitmin

	velo = 500./float(otwsplit)

	res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
	force = res[0]
	power = res[3]
	ratio = res[2]
	try:
	    reserg = rowingphysics.constantwatt_erg(power,r,erg,theconst=1.0,
						    aantal=20,aantal2=20,
						    ratiomin=0.4,ratiomax=0.6)
	except:
	    ergpower = 50.

	ergpower = reserg[4]
	ergvelo = (ergpower/2.8)**(1./3.)
	ergpace = 500./ergvelo
	print((format_pace(otwsplit),format_pace(ergpace)))
	ergsplit110kg.append(ergpace)


    # making the plot

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1)
    ax.plot(ergtime,fifty,color='k')
    ax.scatter(ergsplit50kg,fifty,color='k')
    ax.plot(ergtime,seventy,color='r')
    ax.scatter(ergsplit70kg,seventy,color='r')
    ax.plot(ergtime,ninety,color='g')
    ax.scatter(ergsplit90kg,ninety,color='g')
    ax.plot(ergtime,hundredten,color='b')
    ax.scatter(ergsplit110kg,hundredten,color='b')

    ax.axis([155,75,200,80])
    ax.set_xticks(list(range(80,155,10)))
    ax.set_xlabel('Erg split')
    ax.set_ylabel('OTW split')
    ax.set_title(filename)
    ax.set_yticks(list(range(85,195,10)))
    timeTickFormatter = NullFormatter()

    majorTimeFormatter = FuncFormatter(format_time_tick)
    majorLocator = (15*60)
    ax.xaxis.set_major_formatter(majorTimeFormatter)

    majorFormatter = FuncFormatter(format_pace_tick)
    majorLocator = (5)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_major_formatter(majorFormatter)

    ax.legend(['50kg kleshnev','70kg kleshnev','90kg kleshnev','110kg kleshnev',
	       '50kg me','70kg me','90kg me','110kg me'],
	      prop={'size':10},loc=0)

    grid(True)

    fig.show()

def plotinversedata2(filename,r,rg,erg):
    expdata_raw = pd.read_csv(filename)

    ergtime = rawtoseconds(expdata_raw['erg score'])/4.

    fifty = rawtoseconds(expdata_raw['50kg'])/4.
    sixty = rawtoseconds(expdata_raw['60kg'])/4.
    seventy = rawtoseconds(expdata_raw['70kg'])/4.
    eighty = rawtoseconds(expdata_raw['80kg'])/4.
    ninety = rawtoseconds(expdata_raw['90kg'])/4.
    hundred = rawtoseconds(expdata_raw['100kg'])/4.
    hundredten = rawtoseconds(expdata_raw['110kg'])/4.

    ratio = 0.5

    print('50kg')

    r.mc = 50.0
    r.tempo = 30.

    ergsplit50kg = []

    for otwscore in expdata_raw['50kg']:
	print(otwscore)
	mins, secs = splitvalues(otwscore)
	otwsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromotwsplit(otwsplit,expdata_raw['50kg'],50)
	otwsplitmin = int(otwsplit) / 60
	otwsplitsec = otwsplit-60*otwsplitmin

	velo = 500./float(otwsplit)

	res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
	force = res[0]
	power = res[3]
	ratio = res[2]
	try:
	    ergpower = rowingphysics.find_ergpower(power,r,erg,aantal=5,aantal2=5)
	    print((power,ergpower))
	except:
	    ergpower = 50.
	    

	ergvelo = (ergpower/2.8)**(1./3.)
	ergpace = 500./ergvelo
	print((format_pace(otwsplit),format_pace(ergpace)))
	ergsplit50kg.append(ergpace)


    print('70kg')

    r.mc = 70.0
    r.tempo = 30.

    ergsplit70kg = []

    for otwscore in expdata_raw['70kg']:
	print(otwscore)
	mins, secs = splitvalues(otwscore)
	otwsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromotwsplit(otwsplit,expdata_raw['70kg'],70)
	otwsplitmin = int(otwsplit) / 60
	otwsplitsec = otwsplit-60*otwsplitmin

	velo = 500./float(otwsplit)

	res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
	force = res[0]
	power = res[3]
	ratio = res[2]
	try:
	    ergpower = rowingphysics.find_ergpower(power,r,erg,aantal=5,aantal2=5)
	    print((power,ergpower))


	except:
	    ergpower = 50.

	ergvelo = (ergpower/2.8)**(1./3.)
	ergpace = 500./ergvelo
	print((format_pace(otwsplit),format_pace(ergpace)))
	ergsplit70kg.append(ergpace)


    print('90kg')

    r.mc = 90.0
    r.tempo = 30.

    ergsplit90kg = []

    for otwscore in expdata_raw['90kg']:
	print(otwscore)
	mins, secs = splitvalues(otwscore)
	otwsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromotwsplit(otwsplit,expdata_raw['90kg'],90)
	otwsplitmin = int(otwsplit) / 60
	otwsplitsec = otwsplit-60*otwsplitmin

	velo = 500./float(otwsplit)

	res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
	force = res[0]
	power = res[3]
	ratio = res[2]
	try:
	    ergpower = rowingphysics.find_ergpower(power,r,erg,aantal=5,aantal2=5)

	except:
	    ergpower = 50.

	ergvelo = (ergpower/2.8)**(1./3.)
	ergpace = 500./ergvelo
	print((format_pace(otwsplit),format_pace(ergpace)))
	ergsplit90kg.append(ergpace)


    print('110kg')

    r.mc = 110.0
    r.tempo = 30.

    ergsplit110kg = []

    for otwscore in expdata_raw['110kg']:
	print(otwscore)
	mins, secs = splitvalues(otwscore)
	otwsplit = (60.0*mins+secs)/4.
	r.tempo = tempofromotwsplit(otwsplit,expdata_raw['110kg'],110)
	otwsplitmin = int(otwsplit) / 60
	otwsplitsec = otwsplit-60*otwsplitmin

	velo = 500./float(otwsplit)

	res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
	force = res[0]
	power = res[3]
	ratio = res[2]
	try:
	    ergpower = rowingphysics.find_ergpower(power,r,erg,aantal=5,aantal2=5)

	except:
	    ergpower = 50.

	ergvelo = (ergpower/2.8)**(1./3.)
	ergpace = 500./ergvelo
	print((format_pace(otwsplit),format_pace(ergpace)))
	ergsplit110kg.append(ergpace)


    # making the plot

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1)
    ax.plot(ergtime,fifty,color='k')
    ax.plot(ergsplit50kg,fifty,color='k')
    ax.plot(ergtime,seventy,color='r')
    ax.plot(ergsplit70kg,seventy,color='r')
    ax.plot(ergtime,ninety,color='g')
    ax.plot(ergsplit90kg,ninety,color='g')
    ax.plot(ergtime,hundredten,color='b')
    ax.plot(ergsplit110kg,hundredten,color='b')

    ax.axis([155,75,200,85])
    ax.set_xticks(list(range(85,155,10)))
    ax.set_xlabel('Erg split')
    ax.set_ylabel('OTW split')
    ax.set_title(filename)
    ax.set_yticks(list(range(85,200,10)))
    timeTickFormatter = NullFormatter()

    majorTimeFormatter = FuncFormatter(format_time_tick)
    majorLocator = (15*60)
    ax.xaxis.set_major_formatter(majorTimeFormatter)

    majorFormatter = FuncFormatter(format_pace_tick)
    majorLocator = (5)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_major_formatter(majorFormatter)

    ax.legend(['50kg','50kg  OTW',
	       '70kg','70kg  OTW',
	       '90kg','90kg  OTW',
	       '110kg','110kg  OTW'	       ],
	      prop={'size':10},loc=0)

    grid(True)

    fig.show()

def test_one(r,rg,pace):
    min = pace[0]
    sec = pace[1]
    p = 60*min+sec
    velo = 500./p
    
    res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
    force = res[0]
    power = res[3]
    ratio = res[2]

    print(power)
    
    res2 = rowingphysics.constantwatt(power,r,rg)
    otwsplit = 500./res2[1]

    return otwsplit,otwsplit/p

def test_two(r,e,pace):
    ergsplitmin = pace[0]
    sec = pace[1]
    p = 60*ergsplitmin+sec
    velo = 500./p
    ratio = 0.5
    
    res = rowingphysics.ergtopower(ergsplitmin,sec,ratio,r,e)
    totalpower = res[0]
    ergpower = res[1]


    print((totalpower,ergpower))
    
    reserg = rowingphysics.constantwatt_erg(totalpower,r,e,theconst=1.0,
					    ratiomin=0.4,ratiomax=0.6,aantal=30,aantal2=30)
    powerergdisplay = reserg[4]
    v = (powerergdisplay/2.8)**(1./3.)

    ergsplit = 500./v

    return ergsplit, ergsplit/p

def test_three(r,rg,e,pace):
    min = pace[0]
    sec = pace[1]
    p = 60*min+sec

    print((format_pace(p)))
    
    velo = 500./p
    
    res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
    force = res[0]
    power = res[3]
    ratio = res[2]

    print((power,ratio))
    
    reserg = rowingphysics.constantwatt_erg(power,r,e,theconst=1.0,ratiomin=0.4,ratiomax=0.6)
    powerergdisplay = reserg[4]

    print(powerergdisplay)

    v = (powerergdisplay/2.8)**(1./3.)

    ergpace = 500./v

    print((format_pace(ergpace)))

    ergsplitmin = ergpace / 60
    sec = ergpace - 60.*ergsplitmin

    res = rowingphysics.ergtopower(ergsplitmin,sec,ratio,r,e)
    totalpower = res[0]
    ergpower = res[1]

    print((totalpower, ergpower))

    res2 = rowingphysics.constantwatt(totalpower,r,rg)

    otwsplit = 500./res2[1]

    print((format_pace(otwsplit)))

def test_four(r,rg,e,pace,ratio=0.5):
    min = pace[0]
    sec = pace[1]
    p = 60*min+sec

    print((format_pace(p)))
    
    velo = 500./p
    
    res = rowingphysics.ergtopower(min,sec,ratio,r,e)
    totalpower = res[0]
    ergpower = res[1]

    print((totalpower, ergpower))

    res2 = rowingphysics.constantwatt(totalpower,r,rg)

    otwsplit = 500./res2[1]
    velo = res2[1]

    print((format_pace(otwsplit)))

    res = rowingphysics.constantvelofast(velo,r,rg,Fmax=600,windv=0)
    force = res[0]
    power = res[3]
    ratio = res[2]

    print((power,ratio))
    
    reserg = rowingphysics.constantwatt_erg(power,r,e,theconst=1.0,ratiomin=0.4,ratiomax=0.6)
    powerergdisplay = reserg[4]

    print(powerergdisplay)

    v = (powerergdisplay/2.8)**(1./3.)

    ergpace = 500./v

    print((format_pace(ergpace)))

def test_five(r,rg,e,ergpace,otwpace,ratio=0.5):
    ergmin = ergpace[0]
    ergsec = ergpace[1]
    ergp = 60*ergmin+ergsec

    otwmin = otwpace[0]
    otwsec = otwpace[1]
    otwp = 60*otwmin+otwsec

    

    tempos = np.array([20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.])
    
    ergpace2 = 0.0*tempos
    otwpace2 = 0.0*tempos

    print((ergp,otwp))

    for i in range(len(tempos)):
	r.tempo = tempos[i]

	try:
	    res = rowingphysics.ergtopower(ergmin,ergsec,ratio,r,e)
	except:
	    res = [0,0]
	
	totalpower = res[0]
	ergpower = res[1]

	print((tempos[i],ergpower,totalpower))

	try:
	    res2 = rowingphysics.constantwatt(totalpower,r,rg)
	except:
	    res2 = [0,0]

	print((res2[3]))
	otwpace2[i] = 500./res2[1]

	try:
	    res3 = rowingphysics.constantvelofast(500./otwp,r,rg,Fmax=600,windv=0)
	except:
	    res3 = [0,0,0,0,0]


	force = res3[0]
	power = res3[3]
	ratio = res3[2]


	try:
	    reserg = rowingphysics.constantwatt_erg(power,r,e,theconst=1.0,timestep=0.01,ratiomin=0.4,ratiomax=0.6)
	#    reserg = rowingphysics.constantwatt_erg(power,r,e,theconst=1.0,aantal=20,aantal2=20,ratiomin=0.2,ratiomax=0.8)
	except:
	    reserg = [0,0,0,0,0]

	powerergdisplay = reserg[4]
    

	v = (powerergdisplay/2.8)**(1./3.)
	if v==0:
	    v=1.0

	print((power, ratio, 500./v))

	ergpace2[i] = 500./v    
	

    # making the plot

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tempos,otwpace2,color='k')
    ax.plot(tempos,ergpace2,color='r')
    ax.plot(tempos,ergp+0.0*ergpace2,color='r')
    ax.plot(tempos,otwp+0.0*ergpace2,color='k')

    ax.axis([18,40,150,85])
    ax.set_xlabel('SPM')
    ax.set_ylabel('split')
    ax.set_yticks(list(range(85,150,10)))

    majorFormatter = FuncFormatter(format_pace_tick)
    majorLocator = (5)
    ax.yaxis.set_major_formatter(majorFormatter)

    ax.legen='OTW'

    grid(True)

    fig.show()
    

def test_six(r,rg,e,ergpace,otwpace,ratio=0.5):
    ergmin = ergpace[0]
    ergsec = ergpace[1]
    ergp = 60*ergmin+ergsec

    otwmin = otwpace[0]
    otwsec = otwpace[1]
    otwp = 60*otwmin+otwsec

    

    tempos = np.array([20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.])
    tempoerg = 0.0*tempos
    
    ergpace2 = 0.0*tempos
    otwpace2 = 0.0*tempos

    print((ergp,otwp))

    for i in range(len(tempos)):
	r.tempo = tempos[i]

	try:
	    res = rowingphysics.ergtopower(ergmin,ergsec,ratio,r,e)
	except:
	    res = [0,0]
	
	totalpower = res[0]
	ergpower = res[1]

	print((tempos[i],ergpower,totalpower))

	try:
	    res2 = rowingphysics.constantwatt(totalpower,r,rg)
	except:
	    res2 = [0,0]

	print((res2[3]))
	otwpace2[i] = 500./res2[1]

	try:
	    res3 = rowingphysics.constantvelofast(500./otwp,r,rg,Fmax=600,windv=0)
	except:
	    res3 = [0,0,0,0,0]


	force = res3[0]
	power = res3[3]
	ratio = res3[2]


	try:
	    reserg = rowingphysics.constantwatt_ergtempo(power,r,e,theconst=1.0,timestep=0.01,ratio=ratio)
	except:
	    reserg = [0,0,0,0,0]

	powerergdisplay = reserg[4]
	tempoerg[i] = reserg[0]
    

	v = (powerergdisplay/2.8)**(1./3.)
	if v==0:
	    v=1.0

	print((power, ratio, 500./v))

	ergpace2[i] = 500./v    
	

    # making the plot

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tempos,otwpace2,color='k')
    ax.plot(tempoerg,ergpace2,color='r')
    ax.plot(tempos,ergp+0.0*ergpace2,color='r')
    ax.plot(tempos,otwp+0.0*ergpace2,color='k')

    ax.axis([18,40,150,80])
    ax.set_xlabel('SPM')
    ax.set_ylabel('split')
    ax.set_yticks(list(range(80,150,10)))

    majorFormatter = FuncFormatter(format_pace_tick)
    majorLocator = (5)
    ax.yaxis.set_major_formatter(majorFormatter)

    ax.legen='OTW'

    grid(True)

    fig.show()
    
