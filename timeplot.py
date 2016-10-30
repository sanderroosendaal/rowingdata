#!/Users/gregorysmith/anaconda/bin/python
import numpy
import matplotlib
from pylab import *
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from sys import argv
from matplotlib.ticker import MultipleLocator,FuncFormatter

matplotlib.interactive(False)

def format_pace_tick(x,pos=None):
	min=int(x/60)
	sec=int(x-min*60.)
	sec_str=str(sec).zfill(2)
	template='%d:%s'
	return template % (min,sec_str)

def format_time_tick(x,pos=None):
	hour=int(x/3600)
	min=int((x-hour*3600.)/60)
	min_str=str(min).zfill(2)
	template='%d:%s'
	return template % (hour,min_str)

ut2=136
ut1=143
at=157
tr=163
an=178
max=185

script = argv[0]
readFile = argv[1]

# read the painsled stroke file into a pandas datframe
sled_df=pd.read_csv(readFile)

# remove the start time from the time stamps
sled_df.ix[:,'TimeStamp (sec)']=sled_df.ix[:,'TimeStamp (sec)']-sled_df.ix[0,'TimeStamp (sec)']

number_of_columns = sled_df.shape[1]
number_of_rows = sled_df.shape[0]

# define an additional data frame that will hold the multiple bar plot data and the hr 
# limit data for the plot
hr_df = DataFrame({'key': sled_df.ix[:,0],
				   'hr_ut2': range(number_of_rows),
				   'hr_ut1': range(number_of_rows),
				   'hr_at': range(number_of_rows),
				   'hr_tr': range(number_of_rows),
				   'hr_an': range(number_of_rows),
				   'hr_max': range(number_of_rows),
				   'lim_ut2': ut2,
				   'lim_ut1': ut1,
				   'lim_at': at,
				   'lim_tr': tr,
				   'lim_an': an,
				   'lim_max': max,
				   })
				   
# merge the two dataframes together
df = pd.merge(sled_df,hr_df,left_on='TimeStamp (sec)',right_on='key')

# The following for loop fills up the data for the multiple bar plot.
# The logic sets the values of all the lists to be zero, unless 
# the hr is in the specific range between two limits (eg between UT2 and UT1)
# in that case the list values are set to the current HR value
# When all of these lists are plotted, it makes the multicolored bar plots I love
for i in range(number_of_rows):
	if df.ix[i,' HRCur (bpm)'] < ut2:
		df.ix[i,'hr_ut2'] = df.ix[i,' HRCur (bpm)']
		df.ix[i,'hr_ut1'] = 0.0
		df.ix[i,'hr_at'] = 0.0
		df.ix[i,'hr_tr'] = 0.0
		df.ix[i,'hr_an'] = 0.0
		df.ix[i,'hr_max'] = 0.0		
	elif df.ix[i,' HRCur (bpm)'] < ut1:
		df.ix[i,'hr_ut2'] = 0.0
		df.ix[i,'hr_ut1'] = df.ix[i,' HRCur (bpm)']
		df.ix[i,'hr_at'] = 0.0
		df.ix[i,'hr_tr'] = 0.0
		df.ix[i,'hr_an'] = 0.0
		df.ix[i,'hr_max'] = 0.0		
	elif df.ix[i,' HRCur (bpm)'] < at:
		df.ix[i,'hr_ut2'] = 0.0
		df.ix[i,'hr_ut1'] = 0.0
		df.ix[i,'hr_at'] = df.ix[i,' HRCur (bpm)']
		df.ix[i,'hr_tr'] = 0.0
		df.ix[i,'hr_an'] = 0.0
		df.ix[i,'hr_max'] = 0.0
	elif df.ix[i,' HRCur (bpm)'] < tr:
		df.ix[i,'hr_ut2'] = 0.0
		df.ix[i,'hr_ut1'] = 0.0
		df.ix[i,'hr_at'] = 0.0
		df.ix[i,'hr_tr'] = df.ix[i,' HRCur (bpm)']
		df.ix[i,'hr_an'] = 0.0
		df.ix[i,'hr_max'] = 0.0
	elif df.ix[i,' HRCur (bpm)'] < an:
		df.ix[i,'hr_ut2'] = 0.0
		df.ix[i,'hr_ut1'] = 0.0
		df.ix[i,'hr_at'] = 0.0
		df.ix[i,'hr_tr'] = 0.0
		df.ix[i,'hr_an'] = df.ix[i,' HRCur (bpm)']
		df.ix[i,'hr_max'] = 0.0
	else: 
		df.ix[i,'hr_ut2'] = 0.0
		df.ix[i,'hr_ut1'] = 0.0
		df.ix[i,'hr_at'] = 0.0
		df.ix[i,'hr_tr'] = 0.0
		df.ix[i,'hr_an'] = 0.0
		df.ix[i,'hr_max'] = df.ix[i,' HRCur (bpm)']

			
fig1 = plt.figure(figsize=(12,10))
fig_title = "Input File:  "+readFile+" --- HR / Pace / Rate / Power"

# First panel, hr
ax1 = fig1.add_subplot(4,1,1)
ax1.bar(df.ix[:,'TimeStamp (sec)'],df.ix[:,'hr_ut2'],color='gray', ec='gray')
ax1.bar(df.ix[:,'TimeStamp (sec)'],df.ix[:,'hr_ut1'],color='y',ec='y')
ax1.bar(df.ix[:,'TimeStamp (sec)'],df.ix[:,'hr_at'],color='g',ec='g')
ax1.bar(df.ix[:,'TimeStamp (sec)'],df.ix[:,'hr_tr'],color='blue',ec='blue')
ax1.bar(df.ix[:,'TimeStamp (sec)'],df.ix[:,'hr_an'],color='violet',ec='violet')
ax1.bar(df.ix[:,'TimeStamp (sec)'],df.ix[:,'hr_max'],color='r',ec='r')
ax1.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,'lim_ut2'],color='k')
ax1.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,'lim_ut1'],color='k')
ax1.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,'lim_at'],color='k')
ax1.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,'lim_tr'],color='k')
ax1.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,'lim_an'],color='k')
ax1.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,'lim_max'],color='k')
ax1.text(5,ut2+1.5,"UT2",size=8)
ax1.text(5,ut1+1.5,"UT1",size=8)
ax1.text(5,at+1.5,"AT",size=8)
ax1.text(5,tr+1.5,"TR",size=8)
ax1.text(5,an+1.5,"AN",size=8)
ax1.text(5,max+1.5,"MAX",size=8)

end_time = int(df.ix[df.shape[0]-1,0])
ax1.axis([0,end_time,100,200])
ax1.set_xticks(range(0,end_time,300))
ax1.set_ylabel('BPM')
ax1.set_yticks(range(110,190,10))
ax1.set_title(fig_title)
timeTickFormatter = NullFormatter()
ax1.xaxis.set_major_formatter(timeTickFormatter)

grid(True)

# Second Panel, Pace
ax2 = fig1.add_subplot(4,1,2)
ax2.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' Stroke500mPace (sec/500m)'])
end_time = int(df.ix[df.shape[0]-1,0])
ax2.axis([0,end_time,150,90])
ax2.set_xticks(range(0,end_time,300))
ax2.set_ylabel('(sec/500)')
ax2.set_yticks(range(145,90,-5))
# ax2.set_title('Pace')
grid(True)
majorFormatter = FuncFormatter(format_pace_tick)
majorLocator = (5)
ax2.xaxis.set_major_formatter(timeTickFormatter)
ax2.yaxis.set_major_formatter(majorFormatter)

# Third Panel, rate
ax3 = fig1.add_subplot(4,1,3)
ax3.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' Cadence (stokes/min)'])
rate_ewma = pd.ewma
ax3.axis([0,end_time,14,40])
ax3.set_xticks(range(0,end_time,300))
ax3.set_xlabel('Time (sec)')
ax3.set_ylabel('SPM')
ax3.set_yticks(range(16,40,2))
# ax3.set_title('Rate')
ax3.xaxis.set_major_formatter(timeTickFormatter)
grid(True)

# Fourth Panel, watts
ax4 = fig1.add_subplot(4,1,4)
ax4.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' Power (watts)'])
ax4.axis([0,end_time,100,500])
ax4.set_xticks(range(0,end_time,300))
ax4.set_xlabel('Time (h:m)')
ax4.set_ylabel('Watts')
ax4.set_yticks(range(150,450,50))
# ax4.set_title('Power')
grid(True)
majorTimeFormatter = FuncFormatter(format_time_tick)
majorLocator = (15*60)
ax4.xaxis.set_major_formatter(majorTimeFormatter)

plt.subplots_adjust(hspace=0)

fig2 = plt.figure(figsize=(12,10))
fig_title = "Input File:  "+readFile+" --- Stroke Metrics"

# Top plot is pace
ax5 = fig2.add_subplot(4,1,1)
ax5.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' Stroke500mPace (sec/500m)'])
end_time = int(df.ix[df.shape[0]-1,0])
ax5.axis([0,end_time,150,90])
ax5.set_xticks(range(0,end_time,300))
ax5.set_ylabel('(sec/500)')
ax5.set_yticks(range(145,90,-5))
grid(True)
ax5.set_title(fig_title)
majorFormatter = FuncFormatter(format_pace_tick)
majorLocator = (5)
ax5.xaxis.set_major_formatter(timeTickFormatter)
ax5.yaxis.set_major_formatter(majorFormatter)

# next we plot the drive length
ax6 = fig2.add_subplot(4,1,2)
ax6.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' DriveLength (meters)'])
ax6.axis([0,end_time,1.3,1.6])
ax6.set_xticks(range(0,end_time,300))
ax6.set_xlabel('Time (sec)')
ax6.set_ylabel('Drive Len(m)')
ax6.set_yticks(arange(1.35,1.6,0.05))
ax6.xaxis.set_major_formatter(timeTickFormatter)
grid(True)

# next we plot the drive time and recovery time
ax7 = fig2.add_subplot(4,1,3)
ax7.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' DriveTime (ms)']/1000.)
ax7.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' StrokeRecoveryTime (ms)']/1000.)
ax7.axis([0,end_time,0.0,3.0])
ax7.set_xticks(range(0,end_time,300))
ax7.set_xlabel('Time (sec)')
ax7.set_ylabel('Drv / Rcv Time (s)')
ax7.set_yticks(arange(0.2,3.0,0.2))
ax7.xaxis.set_major_formatter(timeTickFormatter)
grid(True)

# Peak and average force
ax8 = fig2.add_subplot(4,1,4)
ax8.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' AverageDriveForce (lbs)'])
ax8.plot(df.ix[:,'TimeStamp (sec)'],df.ix[:,' PeakDriveForce (lbs)'])
ax8.axis([0,end_time,0,300])
ax8.set_xticks(range(0,end_time,300))
ax8.set_xlabel('Time (h:m)')
ax8.set_ylabel('Force (lbs)')
ax8.set_yticks(range(25,300,25))
# ax4.set_title('Power')
grid(True)
majorTimeFormatter = FuncFormatter(format_time_tick)
majorLocator = (15*60)
ax8.xaxis.set_major_formatter(majorTimeFormatter)


plt.subplots_adjust(hspace=0)

plt.show()
print "done"

