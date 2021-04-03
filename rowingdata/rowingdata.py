# pylint: disable=C0103, C0303, C0325, C0413, W0403, W0611

from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
from six.moves import input

__version__ = "3.3.0"

from collections import Counter

from matplotlib import figure
import matplotlib
try:
    matplotlib.use('TkCairo')
except (ValueError,ImportError):
    matplotlib.use('Agg')

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

from matplotlib.pyplot import grid
from matplotlib.ticker import FuncFormatter, NullFormatter
import shutil
from scipy.signal import savgol_filter

try:
    from six.moves.tkinter import Tk
    tkavail = 1
except ImportError:
    tkavail = 0

import datetime
import getpass

import math
from math import cos

import pickle

import time
import warnings
import sys
if sys.version_info < (3,):
    warnings.warn(
        """You are using master of 'rowingdata' with Python 2.
        Rowingdata will soon be Python 3 only.""",
                  UserWarning)

from sys import platform as _platform

import arrow

import numpy as np
from numpy import isinf, isnan

import pandas as pd
from pandas import DataFrame, Series


from fitparse import FitFile


from scipy import integrate
from scipy.interpolate import griddata

from tqdm import tqdm
weknowphysics = 0

try:
    import rowingphysics
    weknowphysics = 1
    theerg = rowingphysics.erg()
except ImportError:
    weknowphysics = 0

try:
    from . import gpxwrite
    from . import trainingparser
    from . import writetcx
except (ValueError,ImportError):
    import rowingdata.gpxwrite
    import rowingdata.trainingparser
    import rowingdata.writetcx

import requests

try:
    from . import checkdatafiles
except (ValueError,ImportError):
    import rowingdata.checkdatafiles

try:
    from .csvparsers import (
        BoatCoachAdvancedParser, BoatCoachOTWParser,
        RitmoTimeParser, HumonParser,
        BoatCoachParser, CoxMateParser, CSVParser,
        ErgDataParser, ErgStickParser, KinoMapParser,
        MysteryParser, RowPerfectParser, RowProParser,
        QuiskeParser,ETHParser,
        SpeedCoach2Parser, get_empower_rigging, get_file_line,
        get_file_type, get_rowpro_footer, lbstoN,
        make_cumvalues, make_cumvalues_array,
        painsledDesktopParser, skip_variable_footer,
        skip_variable_header, speedcoachParser, timestrtosecs,
        timestrtosecs2, totimestamp, empower_bug_correction,
        get_empower_firmware, NKLiNKLogbookParser,
    )

    from .otherparsers import TCXParser as TCXParserNoHR
    from .otherparsers import (
        FITParser, FitSummaryData, fitsummarydata,TCXParser,
        ExcelTemplate,GPXParser
    )

    from .utils import (
        ewmovingaverage, geo_distance, totimestamp, format_pace,
        format_time, wavg
    )
except (ValueError,ImportError):
    from rowingdata.csvparsers import (
        BoatCoachAdvancedParser, BoatCoachOTWParser,
        RitmoTimeParser, HumonParser,ETHParser,
        BoatCoachParser, CoxMateParser, CSVParser,
        ErgDataParser, ErgStickParser, KinoMapParser,
        MysteryParser, RowPerfectParser, RowProParser,
        QuiskeParser,
        SpeedCoach2Parser, get_empower_rigging, get_file_line,
        get_file_type, get_rowpro_footer, lbstoN,
        make_cumvalues, make_cumvalues_array,
        painsledDesktopParser, skip_variable_footer,
        skip_variable_header, speedcoachParser, timestrtosecs,
        timestrtosecs2, totimestamp, empower_bug_correction,
        get_empower_firmware
    )

    from rowingdata.otherparsers import TCXParser as TCXParserNoHR
    from rowingdata.otherparsers import (
        FITParser, FitSummaryData, fitsummarydata,TCXParser,
        ExcelTemplate
    )

    from rowingdata.utils import (
        ewmovingaverage, geo_distance, totimestamp, format_pace,
        format_time, wavg
    )

if tkavail == 0:
    matplotlib.use('Agg')




def main():
    str = "Executing rowingdata version %s. " % __version__
    if weknowphysics:
        str += rowingphysics.main()
    return str

def my_autopct(pct, cutoff=5):
    return ('%4.1f%%' % pct) if pct > cutoff else ''

def nanstozero(nr):
    if isnan(nr) or isinf(nr):
        return 0
    else:
        return nr


def post_progress(secret,progressurl,progress):
    post_data = {
        "secret":secret,
        "value":progress,
    }

    try:
        s = requests.post(progressurl, data=post_data)
    except:
        return 408
    return s.status_code


# toekomstmuziek - nog niet gebruikt
def make_subplot(ax,r,df,param,mode=['distance','ote'],bars=None,barnames=None):
    if 'distance' in mode:
        xcolumn = 'cum_dist'
        dist_max = 1000
        dist_tick = 100
    else:
        xcolumn = 'TimeStamp (sec)'
        dist_max = 300
        dist_tick = 30

    end_dist = int(df.loc[:, xcolumn].iloc[df.shape[0] - 1]) # replaced ix with loc/iloc

    ax.plot(df.loc[:,xcolumn], df.loc[:,param])

    if bars:
        if barnames is None:
            barnames = ['hr_ut2','hr_ut1','hr_at','hr_tr','hr_an','hr_max']
        if barlimits is None:
            barlimits = ['lim_ut2','lim_ut1','lim_at','lim_tr','lim_an','lim_max']
        if barverbosenames is None:
            barverbosenames = list(self.rwr.hrzones)

    colors = ['gray','y','g','blue','violet','r']

    for i in range(len(bars)):
        ax.bar(df.loc[:,xcolumn],df.loc[:,barnames[i]],
               width=dist_increments,
               color=colors[i],ec=colors[i])
        ax.plot(df.loc[:, xcolumn], df.loc[:, barlimits[i]], color='k')
        ax.text(5,df[barlimits[i]].mean()+1.5,barverbosenames[i],size=8)

    if 'ote' in mode and param == ' Stroke500mPace (sec/500m)':
        yrange = y_axis_range(df.loc[:, param],
                              ultimate=[85, 160], quantiles=[0, 0.9])
        ax.set_ylabel('(/500m)')
        grid(True)
        majorTickformatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
    elif param == ' Stroke500mPace (sec/500m)':
        yrange = y_axis_range(df.loc[:, param],
                              ultimate=[85, 240], quantiles=[0, 0.9])
        majorTickformatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        ax.set_ylabel('(/500m)')
    elif param == ' Cadence (stokes/min)':
        ax.set_ylabel('SPM')
        ax.set_yticks(list(range(16,40,2)))
    elif param == ' DriveLength (meters)':
        yrange = y_axis_range(df.loc[:,param],
                              ultimate=[1.0,15])
        ax.set_ylabel('Drive Length (m)')
    elif param == ' Power (watts)':
        yrange = y_axis_range(df.loc[:, param],
                              ultimate=[0,555],miny=0)
        ax.set_ylabel('Power (Watts)')


    ax.grid(True,which='major',axis='both')


    xTickFormatter = NullFormatter()
    if 'last' in mode:
        if 'time' in mode:
            xTickFormatter = FuncFormatter(format_time_tick)
            majorLocator = (15 * 60)
            if end_dist < dist_max:
                majorLocator = (1 * 60)
        else:
            xTickFormatter = FuncFormatter(format_dist_tick)
            majorLocator = (1000)


    ax.yaxis.set_major_formatter(majorTickFormatter)
    if 'time' in mode:
        ax.set_xlabel('Time (sec)')
    else:
        ax.set_axlabel('Distance (m)')

    if end_dist < dist_max:
        ax.set_xticks(list(range(dist_tick, end_dist, dist_tick)))

    ax.xaxis.set_major_formatter(xTickFormatter)




def make_hr_bars(ax1,r,df,mode=['distance'],title=None,gridtrue=True,axis='x'):

    if not title:
        fig_title = "Input File:  " + r.readfilename + " --- HR / Pace / Rate / Power"
        if r.dragfactor:
            fig_title += " Drag %d" % r.dragfactor
    else:
        fig_title=title

    if 'distance' in mode:
        xcolumn = 'cum_dist'

        dist_increments = df.loc[:, xcolumn].diff() # replaced ix with loc
        dist_increments[0] = dist_increments[1]
        dist_max = 1000
        dist_tick = 100
    else:
        xcolumn = 'TimeStamp (sec)'
        dist_increments = df.loc[:, xcolumn].diff()
        dist_increments[0] = dist_increments[1]
        dist_increments = 0.5*(abs(dist_increments)+(dist_increments))

        dist_max = 300
        dist_tick = 30


    end_dist = int(df.loc[:, xcolumn].iloc[df.shape[0] - 1]) # replaced ix with loc/iloc


    df.loc[:,'tempval'] = df.loc[:,' HRCur (bpm)']
    df.loc[df.hr_ut2==0,'tempval']=0
    ax1.fill_between(df.loc[:,xcolumn], df.tempval,color='gray')

    df.loc[:,'tempval'] = df.loc[:,' HRCur (bpm)']
    df.loc[df.hr_ut1==0,'tempval']=0
    ax1.fill_between(df.loc[:,xcolumn], df.tempval,color='y')

    df.loc[:,'tempval'] = df.loc[:,' HRCur (bpm)']
    df.loc[df.hr_at==0,'tempval']=0
    ax1.fill_between(df.loc[:,xcolumn], df.tempval,color='g')

    df.loc[:,'tempval'] = df.loc[:,' HRCur (bpm)']
    df.loc[df.hr_tr==0,'tempval']=0
    ax1.fill_between(df.loc[:,xcolumn], df.tempval,color='blue')

    df.loc[:,'tempval'] = df.loc[:,' HRCur (bpm)']
    df.loc[df.hr_an==0,'tempval']=0
    ax1.fill_between(df.loc[:,xcolumn], df.tempval,color='violet')

    df.loc[:,'tempval'] = df.loc[:,' HRCur (bpm)']
    df.loc[df.hr_max==0,'tempval']=0
    ax1.fill_between(df.loc[:,xcolumn], df.tempval,color='r')


    ax1.plot(df.loc[:, xcolumn], df.loc[:, 'lim_ut2'], color='k')
    ax1.plot(df.loc[:, xcolumn], df.loc[:, 'lim_ut1'], color='k')
    ax1.plot(df.loc[:, xcolumn], df.loc[:, 'lim_at'], color='k')
    ax1.plot(df.loc[:, xcolumn], df.loc[:, 'lim_tr'], color='k')
    ax1.plot(df.loc[:, xcolumn], df.loc[:, 'lim_an'], color='k')
    ax1.plot(df.loc[:, xcolumn], df.loc[:, 'lim_max'], color='k')

    ax1.text(5, r.rwr.ut2 + 1.5, r.rwr.hrzones[1], size=8)
    ax1.text(5, r.rwr.ut1 + 1.5, r.rwr.hrzones[2], size=8)
    ax1.text(5, r.rwr.at + 1.5, r.rwr.hrzones[3], size=8)
    ax1.text(5, r.rwr.tr + 1.5, r.rwr.hrzones[4], size=8)
    ax1.text(5, r.rwr.an + 1.5, r.rwr.hrzones[5], size=8)
    ax1.text(5, r.rwr.max + 1.5, r.rwr.hrzones[6], size=8)


    ax1.axis([0, end_dist, 100, 1.1 * r.rwr.max])
    ax1.set_xticks(list(range(dist_max, end_dist, dist_max)))
    if end_dist < dist_max:
        ax1.set_xticks(list(range(dist_tick, end_dist, dist_tick)))
    ax1.set_ylabel('BPM')
    ax1.set_yticks(list(range(110, 200, 10)))
    ax1.set_title(fig_title)
    if 'time' in mode:
        timeTickFormatter = NullFormatter()
        ax1.xaxis.set_major_formatter(timeTickFormatter)

    ax1.grid(gridtrue,which='major',axis=axis)



def make_pace_plot(ax2,r,df,mode=['distance','ote'],pacerange=[],axis='both',gridtrue=True):
    if 'distance' in mode:
        xcolumn = 'cum_dist'
        dist_max = 1000
        dist_tick = 100
    else:
        xcolumn = 'TimeStamp (sec)'
        dist_max = 300
        dist_tick = 30

    end_dist = int(df.loc[:, xcolumn].iloc[df.shape[0] - 1]) # replaced ix with loc/iloc

    ax2.plot(df.loc[:, xcolumn], df.loc[:, ' Stroke500mPace (sec/500m)'])
    if 'wind' in mode:
        try:
            ax2.plot(df.loc[:, xcolumn],df.loc[:, 'nowindpace'])
            ax2.legend(['Pace', 'Wind corrected pace'],
                       prop={'size':10}, loc=0)
        except KeyError:
            pass
    if 'ote' in mode:
        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=[85, 160], quantiles=[0, 0.9])
    else:
        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=[85, 240], quantiles=[0, 0.9])

    if len(pacerange) == 2:
        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=pacerange,quantiles=[0,0.9])

    try:
        ax2.axis([0, end_dist, yrange[1], yrange[0]])
    except ValueError:
        ax2.axis([0, end_dist, 85, 240])

    ax2.set_xticks(list(range(dist_max, end_dist, dist_max)))
    if end_dist < dist_max:
        ax2.set_xticks(list(range(dist_tick, end_dist, dist_tick)))
    ax2.set_ylabel('(/500)')
    #       ax2.set_yticks(range(145,95,-5))
    # grid(True)
    majorTickFormatter = FuncFormatter(format_pace_tick)
    majorLocator = (5)
    ax2.yaxis.set_major_formatter(majorTickFormatter)
    ax2.grid(gridtrue,which='major',axis=axis)
    if 'time' in mode:
        timeTickFormatter = NullFormatter()
        ax2.xaxis.set_major_formatter(timeTickFormatter)

def make_spm_plot(ax3,r,df,mode=['distance'],axis='both',gridtrue=True):
    if 'distance' in mode:
        xcolumn = 'cum_dist'
        dist_max = 1000
        dist_tick = 100
    else:
        xcolumn = 'TimeStamp (sec)'
        dist_max = 300
        dist_tick = 100

    end_dist = int(df.loc[:, xcolumn].iloc[df.shape[0] - 1]) # replaced ix with loc/iloc
    ax3.plot(df.loc[:, xcolumn], df.loc[:, ' Cadence (stokes/min)'])
    ax3.axis([0, end_dist, 14, 40])
    ax3.set_xticks(list(range(dist_max, end_dist, dist_max)))
    if end_dist < dist_max:
        ax3.set_xticks(list(range(dist_tick, end_dist, dist_tick)))
    ax3.set_ylabel('SPM')
    ax3.set_yticks(list(range(16, 40, 2)))
    if 'time' in mode:
        if 'last' in mode:
            timeTickFormatter = FuncFormatter(format_time_tick)
        else:
            timeTickFormatter = NullFormatter()
        ax3.xaxis.set_major_formatter(timeTickFormatter)

    ax3.grid(gridtrue,which='major',axis=axis)

def make_drivelength_plot(ax6,r,df,mode=['distance'],axis='both',gridtrue=True):
    if 'distance' in mode:
        xcolumn = 'cum_dist'
        dist_max = 1000
        dist_tick = 100
    else:
        xcolumn = 'TimeStamp (sec)'
        dist_max = 300
        dist_tick = 100

    end_dist = int(df.loc[:, xcolumn].iloc[df.shape[0] - 1]) # replaced ix with loc/iloc
    ax6.plot(df.loc[:, xcolumn],
             df.loc[:, ' DriveLength (meters)'])
    yrange = y_axis_range(df.loc[:, ' DriveLength (meters)'],
                          ultimate=[1.0, 15])
    ax6.axis([0, end_dist, yrange[0], yrange[1]])
    ax6.set_xticks(list(range(0, end_dist, dist_max)))
    if end_dist < dist_max:
        ax6.set_xticks(list(range(dist_tick, end_dist, dist_tick)))
    ax6.set_xlabel('Time (sec)')
    ax6.set_ylabel('Drive Len(m)')
    #       ax6.set_yticks(np.arange(1.35,1.6,0.05))
    if 'time' in mode:
        timeTickFormatter = NullFormatter()
        ax6.xaxis.set_major_formatter(timeTickFormatter)
    ax6.grid(gridtrue,which='major',axis=axis)

def make_drivetime_plot(ax7,self,df,mode=['distance'],axis='both',gridtrue=True):
    if 'distance' in mode:
        xcolumn = 'cum_dist'
        dist_max = 1000
        dist_tick = 100
    else:
        xcolumn = 'TimeStamp (sec)'
        dist_max = 300
        dist_tick = 100

    end_dist = int(df.loc[:, xcolumn].iloc[df.shape[0] - 1]) # replaced ix with loc/iloc
    ax7.plot(df.loc[:, xcolumn],
             df.loc[:, ' DriveTime (ms)'] / 1000.)
    ax7.plot(df.loc[:, xcolumn],
             df.loc[:, ' StrokeRecoveryTime (ms)'] / 1000.)
    s = np.concatenate((df.loc[:, ' DriveTime (ms)'].values / 1000.,
                        df.loc[:, ' StrokeRecoveryTime (ms)'].values / 1000.))
    yrange = y_axis_range(s, ultimate=[0.5, 4])

    ax7.axis([0, end_dist, yrange[0], yrange[1]])
    ax7.set_xticks(list(range(0, end_dist, dist_max)))
    if end_dist < dist_max:
        ax7.set_xticks(list(range(dist_tick, end_dist, dist_tick)))

    ax7.set_xlabel('Time (sec)')
    ax7.set_ylabel('Drv / Rcv Time (s)')
    #       ax7.set_yticks(np.arange(0.2,3.0,0.2))
    if 'time' in mode:
        timeTickFormatter = NullFormatter()
        ax7.xaxis.set_major_formatter(timeTickFormatter)
    ax7.grid(gridtrue,which='major',axis=axis)

def make_force_plot(ax8,self,df,mode=['distance'],axis='both',gridtrue=True):
    if 'distance' in mode:
        xcolumn = 'cum_dist'
        dist_max = 1000
        dist_tick = 100
    else:
        xcolumn = 'TimeStamp (sec)'
        dist_max = 300
        dist_tick = 100

    end_dist = int(df.loc[:, xcolumn].iloc[df.shape[0] - 1]) # replaced ix with loc/iloc
    ax8.plot(df.loc[:, xcolumn],
             df.loc[:, ' AverageDriveForce (lbs)'] * lbstoN)
    ax8.plot(df.loc[:, xcolumn],
             df.loc[:, ' PeakDriveForce (lbs)'] * lbstoN)
    s = np.concatenate((df.loc[:, ' AverageDriveForce (lbs)'].values * lbstoN,
                        df.loc[:, ' PeakDriveForce (lbs)'].values * lbstoN))
    yrange = y_axis_range(s, ultimate=[0, 1000])

    ax8.axis([0, end_dist, yrange[0], yrange[1]])
    ax8.set_xticks(list(range(0, end_dist, dist_max)))
    if end_dist < dist_max:
        ax8.set_xticks(list(range(dist_tick, end_dist, dist_tick)))

    if 'distance' in mode:
        ax8.set_xlabel('Dist (m)')
    else:
        ax8.set_xlabel('Time (h:m)')

    ax8.set_ylabel('Force (N)')
    #       ax8.set_yticks(range(25,300,25))
    # ax4.set_title('Power')
    ax8.grid(gridtrue,which='major',axis=axis)

    if 'time' in mode:
        timeTickFormatter = FuncFormatter(format_time_tick)
        majorLocator = (15 * 60)
        if end_dist < dist_max:
            majorLocator = (1 * 60)
        ax8.xaxis.set_major_formatter(timeTickFormatter)
    else:
        majorKmFormatter = FuncFormatter(format_dist_tick)
        majorLocator = (1000)
        ax8.xaxis.set_major_formatter(majorKmFormatter)



def make_power_plot(ax4,r,df,mode=['distance'],axis='both',gridtrue=True):
    if 'distance' in mode:
        xcolumn = 'cum_dist'

        dist_increments = df.loc[:, xcolumn].diff() # replaced ix with loc
        dist_increments[0] = dist_increments[1]
        dist_max = 1000
        dist_tick = 100
    else:
        xcolumn = 'TimeStamp (sec)'
        dist_increments = df.loc[:, xcolumn].diff()
        dist_increments[0] = dist_increments[1]
        dist_increments = 0.5*(abs(dist_increments)+(dist_increments))
        dist_max = 300
        dist_tick = 30



    end_dist = int(df.loc[:, xcolumn].iloc[df.shape[0] - 1]) # replaced ix with loc/iloc

    df.loc[:,'tempval'] = df.loc[:,' Power (watts)']
    df.loc[df.pw_ut2==0,'tempval']=0
    ax4.fill_between(df.loc[:,xcolumn], df.tempval,color='gray')

    df.loc[:,'tempval'] = df.loc[:,' Power (watts)']
    df.loc[df.pw_ut1==0,'tempval']=0
    ax4.fill_between(df.loc[:,xcolumn], df.tempval,color='y')

    df.loc[:,'tempval'] = df.loc[:,' Power (watts)']
    df.loc[df.pw_at==0,'tempval']=0
    ax4.fill_between(df.loc[:,xcolumn], df.tempval,color='g')

    df.loc[:,'tempval'] = df.loc[:,' Power (watts)']
    df.loc[df.pw_tr==0,'tempval']=0
    ax4.fill_between(df.loc[:,xcolumn], df.tempval,color='blue')

    df.loc[:,'tempval'] = df.loc[:,' Power (watts)']
    df.loc[df.pw_an==0,'tempval']=0
    ax4.fill_between(df.loc[:,xcolumn], df.tempval,color='violet')

    df.loc[:,'tempval'] = df.loc[:,' Power (watts)']
    df.loc[df.pw_max==0,'tempval']=0
    ax4.fill_between(df.loc[:,xcolumn], df.tempval,color='r')


    ax4.plot(df.loc[:, xcolumn], df.loc[:, 'limpw_ut2'], color='k')
    ax4.plot(df.loc[:, xcolumn], df.loc[:, 'limpw_ut1'], color='k')
    ax4.plot(df.loc[:, xcolumn], df.loc[:, 'limpw_at'], color='k')
    ax4.plot(df.loc[:, xcolumn], df.loc[:, 'limpw_tr'], color='k')
    ax4.plot(df.loc[:, xcolumn], df.loc[:, 'limpw_an'], color='k')

    end_dist = int(df.loc[df.index[-1], xcolumn])

    yrange = y_axis_range(df.loc[:, ' Power (watts)'],
                          ultimate=[0, 555], miny=0)
    ax4.axis([0, end_dist, yrange[0], yrange[1]])

    ut2, ut1, at, tr, an = r.rwr.ftp * \
                    np.array(r.rwr.powerperc) / 100.

    if ut2 + 1.5 < yrange[1] and ut2 + 1.5 > yrange[0]:
        ax4.text(5, ut2 + 1.5, r.rwr.powerzones[1], size=8)
    if ut1 + 1.5 < yrange[1] and ut1 + 1.5 > yrange[0]:
        ax4.text(5, ut1 + 1.5, r.rwr.powerzones[2], size=8)
    if at + 1.5 < yrange[1] and at + 1.5 > yrange[0]:
        ax4.text(5, at + 1.5, r.rwr.powerzones[3], size=8)
    if tr + 1.5 < yrange[1] and tr + 1.5 > yrange[0]:
        ax4.text(5, tr + 1.5, r.rwr.powerzones[4], size=8)
    if an + 1.5 < yrange[1] and an + 1.5 > yrange[0]:
        ax4.text(5, an + 1.5, r.rwr.powerzones[5], size=8)

    ax4.set_xticks(list(range(0, end_dist, dist_max)))
    if end_dist < dist_max:
        ax4.set_xticks(list(range(0, end_dist, dist_tick)))
    if 'distance' in mode:
        ax4.set_xlabel('Dist (m)')
    else:
        ax4.set_xlabel('Time (h:m)')
    ax4.set_ylabel('Power (Watts)')

        #       ax4.set_yticks(range(110,200,10))

    if 'time' in mode:
        timeTickFormatter = FuncFormatter(format_time_tick)
        majorLocator = (15 * 60)
        if end_dist < dist_max:
            majorLocator = (1 * 60)
        ax4.xaxis.set_major_formatter(timeTickFormatter)
    else:
        majorKmFormatter = FuncFormatter(format_dist_tick)
        majorLocator = (1000)
        ax4.xaxis.set_major_formatter(majorKmFormatter)

    ax4.grid(gridtrue,which='major',axis=axis)

def tailwind(bearing, vwind, winddir, vstream=0):
    """ Calculates head-on head/tailwind in direction of rowing

    positive numbers are tail wind

    """

    b = math.radians(bearing)
    w = math.radians(winddir)

    vtail = -vwind * cos(w - b) - vstream

    return vtail

def copytocb(s):
    """ Copy to clipboard for pasting into blog

    Doesn't work on Mac OS X
    """
    if (_platform == 'win32'):
        r = Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(s)
        r.destroy
        print("Summary copied to clipboard")

    else:
        res = "Your platform {pl} is not supported".format(
            pl=_platform
        )
        print(res)

def phys_getpower(velo, rower, rigging,
                  bearing, vwind, winddirection, vstream=0):
    power = 0
    tw = tailwind(bearing, vwind, winddirection, vstream=0)
    velowater = velo - vstream
    if (weknowphysics == 1):
        res = rowingphysics.constantvelofast(velowater, rower, rigging, Fmax=600,
                                             windv=tw)
        force = res[0]
        power = res[3]
        ratio = res[2]
        res2 = rowingphysics.constantwattfast(
            power, rower, rigging, Fmax=600, windv=0)
        vnowind = res2[1]
        pnowind = 500. / res2[1]
        if (power > 100):
            try:
                reserg = rowingphysics.constantwatt_erg(power, rower,
                                                        theerg, theconst=1.0,
                                                        aantal=20, aantal2=20,
                                                        ratiomin=ratio - 0.2, ratiomax=ratio + 0.2)
            except:
                reserg = [np.nan, np.nan, np.nan, np.nan, np.nan]
        else:
            reserg = [np.nan, np.nan, np.nan, np.nan, np.nan]
        ergpower = reserg[4]

        result = [power, ratio, force, pnowind, ergpower]
    else:
        result = [np.nan, np.nan, np.nan, np.nan, np.nan]

    return result


def write_obj(obj, filename):
    """ Save an object (e.g. your rower) to a file
    """
    pickle.dump(obj, open(filename, "wb"))

def read_obj(filename):
    """ Read an object (e.g. your rower, including passwords) from a file
        Usage: john=rowingdata.read_obj("john.txt")
    """
    res = pickle.load(open(filename,'rb'))
    return res

def getrigging(fileName="my1x.txt"):
    """ Read a rigging object
    """

    try:
        rg = pickle.load(open(fileName,'rb'))
    except (IOError, ImportError, ValueError):
        if __name__ == '__main__':
            print("Getrigging: File doesn't exist or is not valid. Creating new")
            print(fileName)
        if (weknowphysics == 1):
            rg = rowingphysics.rigging()
        else:
            rg = 0

    return rg

def getrower(fileName="defaultrower.txt", mc=70.0):
    """ Read a rower object

    """

    try:
        r = pickle.load(open(fileName,'rb'))
    except (IOError, ImportError):
        if __name__ == '__main__':
            print("Getrower: Default rower file doesn't exist. Create new rower")
        r = rower(mc=mc)

    return r




def histodata(rows):
    # calculates Power/Stroke Histo data from a series of rowingdata class rows
    power = np.array([])
    for row in rows:
        power = np.concatenate((power, row.df[' Power (watts)'].values))

    return power



def cumcpdata(rows,debug=False):
    delta = []
    cpvalue = []
    avgpower = {}


    maxt = 10.
    for row in rows:
        tt = row.df[' ElapsedTime (sec)'].copy()
        tt = tt-tt[0]
        thismaxt = tt.max()+10.
        if thismaxt > maxt:
            maxt = thismaxt

    maxlog10 = np.log10(maxt)

    logarr = np.arange(100) * maxlog10 / 100.

    logarr = 10.**(logarr)

    for row in rows:
        tt = row.df[' ElapsedTime (sec)'].copy()
        tt = tt-tt[0]
        ww = row.df[' Power (watts)'].copy()

        tmax = tt.max()
        if debug:
            print('tmax = ',tmax)

        if tmax > 500000:
            newlen = int(tmax/2000.)
            newt = np.arange(newlen)*tmax/float(newlen)
            deltat = newt[1]-newt[0]
        else:
            newt = np.arange(0,tmax,10.)
            deltat = 10.

        ww = griddata(tt.values,
                      ww.values,
                      newt,method='linear',
                      rescale=True)

        tt = pd.Series(newt)
        ww = pd.Series(ww)

        G = pd.Series(ww.cumsum())
        G = pd.concat([pd.Series([0]),G])

        h = np.mgrid[0:len(tt)+1:1,0:len(tt)+1:1]

        distances = pd.DataFrame(h[1]-h[0])

        if debug:
            print(len(tt))
            print(distances)

        ones = 1+np.zeros(len(G))

        Ghor = np.outer(ones,G)
        Gver = np.outer(G,ones)

        Gdif = Ghor - Gver

        Gdif = np.tril(Gdif.T).T

        Gdif = pd.DataFrame(Gdif)

        if debug:
            print(Gdif)

        F = (Gdif)/(distances)

        if debug:
            print('=====================F============')
            print(F)
            print('==================================')


        F.fillna(inplace=True,method='ffill',axis=1)
        F.fillna(inplace=True,value=0)

        restime = []
        power = []

        for i in np.arange(0,len(tt)+1,1):
            restime.append(deltat*i)
            cp = np.diag(F,i).max()
            if debug:
                print(np.diag(F,i))
                print(i,deltat*i,cp)
            power.append(cp)

        power[0] = power[1]

        restime = np.array(restime)
        power = np.array(power)


        #power[0] = power[1]

        cpvalues = griddata(restime,power,
                            logarr,method='linear', fill_value=0)



        for cpv in cpvalues:
            cpvalue.append(cpv)
        for d in logarr:
            delta.append(d)


    velo = (np.array(cpvalue)/2.8)**(1./3.)
    d = np.array(delta)*velo

    df = pd.DataFrame(
        {
            'Delta':delta,
            'CP':cpvalue,
            'Distance':d,

        }
        )

    df = df.sort_values(['Delta', 'CP'], ascending=[1, 0])
    df = df.drop_duplicates(subset='Delta', keep='first')

    #df = df[df['Distance']>100]

    if debug:
        return df, F

    return df




def interval_string(nr, totaldist, totaltime, avgpace, avgspm,
                    avghr, maxhr, avgdps, avgpower,
                    separator='|'):
    """ Used to create a nifty text string with the data for the interval
    """

    try:
        stri = "{nr:0>2.0f}{sep}{td:0>5.0f}{sep}{inttime:0>5}{sep}".format(
            nr=nr,
            sep=separator,
            td=totaldist,
            inttime=format_pace(totaltime)
        )
    except ValueError:
        stri = "{nr}{sep}{td:0>5.0f}{sep}{inttime:0>5}{sep}".format(
            nr=nr,
            sep=separator,
            td=totaldist,
            inttime=format_pace(totaltime)
        )

    stri += "{tpace:0>7}{sep}{tpower:0>5.1f}{sep}{tspm:0>4.1f}{sep}{thr:0>5.1f}".format(
        tpace=format_pace(avgpace),
        sep=separator,
        tspm=avgspm,
        thr=avghr,
        tpower=avgpower,
    )

    stri += "{sep}{tmaxhr:3.1f}{sep}{tdps:0>4.1f}".format(
        sep=separator,
        tmaxhr=maxhr,
        tdps=avgdps
    )

    stri += "\n"
    return stri

def workstring(totaldist, totaltime, avgpace, avgspm, avghr, maxhr, avgdps,
               avgpower,
               separator="|", symbol='W'):

    if np.isnan(totaldist):
        totaldist = 0
    if np.isnan(avgpace):
        avgpace = 0
    if np.isnan(avgspm):
        avgspm = 0
    if np.isnan(avghr):
        avghr = 0
    if np.isnan(maxhr):
        maxhr = 0
    if np.isnan(avgdps):
        avgdps = 0
    if np.isnan(avgpower):
        avgpower = 0
    if np.isnan(totaltime):
        totaltime = 0

    pacestring = format_pace(avgpace)

    stri1 = symbol
    stri1 += "-{sep}{dtot:0>5.0f}{sep}".format(
        sep=separator,
        dtot=totaldist,
        # tottime=format_time(totaltime),
        # pacestring=pacestring
    )

    stri1 += format_time(totaltime) + separator + pacestring

    stri1 += "{sep}{avgpower:0>5.1f}{sep}{avgsr:0>4.1f}{sep}{avghr:0>5.1f}{sep}".format(
        avgsr=avgspm,
        avgpower=avgpower,
        sep=separator,
        avghr=avghr
    )

    stri1 += "{maxhr:0>5.1f}{sep}{avgdps:0>4.1f}\n".format(
        sep=separator,
        maxhr=maxhr,
        avgdps=avgdps
    )

    return stri1


def summarystring(totaldist, totaltime, avgpace, avgspm, avghr, maxhr,
                  avgdps, avgpower,
                  readFile="",
                  separator="|"):
    """ Used to create a nifty string summarizing your entire row
    """

    if np.isnan(totaldist):
        totaldist = 0
    if np.isnan(avgpace):
        avgpace = 0
    if np.isnan(avgspm):
        avgspm = 0
    if np.isnan(avghr):
        avghr = 0
    if np.isnan(maxhr):
        maxhr = 0
    if np.isnan(avgdps):
        avgdps = 0
    if np.isnan(avgpower):
        avgpower = 0
    if np.isnan(totaltime):
        totaltime = 0

    stri1 = "Workout Summary - " + readFile + "\n"
    stri1 += "--{sep}Total{sep}-Total----{sep}--Avg--{sep}-Avg-{sep}Avg-{sep}-Avg-{sep}-Max-{sep}-Avg\n".format(
        sep=separator)
    stri1 += "--{sep}Dist-{sep}-Time-----{sep}-Pace--{sep}-Pwr-{sep}SPM-{sep}-HR--{sep}-HR--{sep}-DPS\n".format(
        sep=separator)

    pacestring = format_pace(avgpace)

    #    stri1 += "--{sep}{dtot:0>5.0f}{sep}{tottime:7.1f}{sep}".format(
    stri1 += "--{sep}{dtot:0>5.0f}{sep}".format(
        sep=separator,
        dtot=totaldist,
        # tottime=format_time(totaltime),
        # pacestring=pacestring
    )

    stri1 += format_time(totaltime) + separator + pacestring

    stri1 += "{sep}{avgpower:0>5.1f}".format(
        sep=separator,
        avgpower=avgpower,
    )

    stri1 += "{sep}{avgsr:2.1f}{sep}{avghr:0>5.1f}{sep}".format(
        avgsr=avgspm,
        sep=separator,
        avghr=avghr
    )

    stri1 += "{maxhr:0>5.1f}{sep}{avgdps:0>4.1f}\n".format(
        sep=separator,
        maxhr=maxhr,
        avgdps=avgdps
    )

    return stri1


def format_pace_tick(x, pos=None):
    min = int(x / 60)
    sec = int(x - min * 60.)
    sec_str = str(sec).zfill(2)
    template = '%d:%s'
    return template % (min, sec_str)


def y_axis_range(ydata, **kwargs):
    # ydata,miny=0,padding=.1,ultimate=[-1e9,1e9]):

    ds = pd.Series(ydata)

    ds = pd.to_numeric(ds,errors='coerce')

    if 'quantiles' in kwargs:
        qmin = kwargs['quantiles'][0]
        qmax = kwargs['quantiles'][1]
    else:
        qmin = 0.01
        qmax = 0.99

    if 'miny' in kwargs:
        ymin = kwargs['miny']
    else:
        ymin = np.ma.masked_invalid(ydata).min()
        ymin = ds.quantile(q=qmin)

    if not 'ultimate' in kwargs:
        ultimate = [-1e9, 1e9]
    else:
        ultimate = kwargs['ultimate']

    if not 'padding' in kwargs:
        padding = .1
    else:
        padding = kwargs['padding']

    # ydata must by a numpy array


    ymax = np.ma.masked_invalid(ydata.astype(float)).max()
    ymax = ds.quantile(q=qmax)

    yrange = ymax - ymin
    yrangemin = ymin
    yrangemax = ymax

    if (yrange == 0):
        if 'miny' not in kwargs:
            if ymin == 0:
                yrangemin = -padding
            else:
                yrangemin = ymin - ymin * padding
        else:
            yrangemin = ymin
        if ymax == 0:
            yrangemax = padding
        else:
            yrangemax = ymax + ymax * padding
    else:
        if 'miny' not in kwargs:
            yrangemin = ymin - padding * yrange
        else:
            yrangemin = ymin

        yrangemax = ymax + padding * yrange

    if (yrangemin < ultimate[0]):
        yrangemin = ultimate[0]

    if (yrangemax > ultimate[1]):
        yrangemax = ultimate[1]

    if not np.isfinite(yrangemin):
        yrangemin = ymin

    if not np.isfinite(yrangemax):
        yrangemax = ymax

    return [yrangemin, yrangemax]


def format_dist_tick(x, pos=None):
    km = x / 1000.
    template = '%6.3f'
    return template % (km)

def format_time_tick(x, pos=None):
    hour = int(x / 3600)
    min = int((x - hour * 3600.) / 60)
    min_str = str(min).zfill(2)
    template = '%d:%s'
    return template % (hour, min_str)


class summarydata:
    """ This is used to create nice summary texts from CrewNerd's summary CSV

    Usage: sumd=rowingdata.summarydata("crewnerdsummary.CSV")

           sumd.allstats()

           sumd.shortstats()

           """

    def __init__(self, readFile):
        self.readFile = readFile
        sumdf = pd.read_csv(readFile, sep=None)
        try:
            sumdf['Strokes']
        except KeyError:
            sumdf = pd.read_csv(readFile, sep=None)
        self.sumdf = sumdf

        # prepare Work Data
        # remove "Just Go"
        #s2=self.sumdf[self.sumdf['Workout Name']<>'Just Go']
        s2 = self.sumdf
        s3 = s2[~s2['Interval Type'].str.contains("Rest")]
        self.workdata = s3

    def allstats(self, separator="|"):

        stri2 = "Workout Details\n"
        stri2 += "#-{sep}SDist{sep}-Split-{sep}-SPace-{sep}SPM-{sep}AvgHR{sep}MaxHR{sep}DPS-\n".format(
            sep=separator
        )

        avghr = self.workdata['Avg HR'].mean()
        avgsr = self.workdata['Avg SR'].mean()
        maxhr = self.workdata['Max HR'].mean()
        maxsr = self.workdata['Max SR'].mean()
        totaldistance = self.workdata['Distance (m)'].sum()
        totalstrokes = self.workdata['Strokes'].sum()

        # min=int(avgpace/60)
        # sec=int(10*(avgpace-min*60.))/10.
        # pacestring=str(min)+":"+str(sec)

        nr_rows = self.workdata.shape[0]

        tothour = 0
        totmin = 0
        totsec = 0
        tottimehr = 0
        tottimespm = 0

        for index,row in self.workdata.iterrows():
            inttime = row['Time']
            thr = row['Avg HR']
            td = row['Distance (m)']
            tpace = row['Avg Pace (/500m)']
            tspm = row['Avg SR']
            tmaxhr = row['Max HR']
            tstrokes = row['Strokes']

            tdps = td / (1.0 * tstrokes)

            try:
                t = datetime.datetime.strptime(inttime, "%H:%M:%S.%f")
            except ValueError:
                try:
                    t = datetime.datetime.strptime(inttime, "%M:%S")
                except ValueError:
                    t = datetime.datetime.strptime(inttime, "%H:%M:%S")

            tothour = tothour + t.hour
            tottimehr += (t.hour * 3600 + t.minute * 60 + t.second) * thr
            tottimespm += (t.hour * 3600 + t.minute * 60 + t.second) * tspm

            totmin = totmin + t.minute
            if (totmin >= 60):
                totmin = totmin - 60
                tothour = tothour + 1

            totsec = totsec + t.second + 0.1 * \
                int(t.microsecond / 1.e5)  # plus tenths
            if (totsec >= 60):
                totsec = totsec - 60
                totmin = totmin + 1

            stri2 += "{nr:0>2}{sep}{td:0>5}{sep} {inttime:0>5} {sep}".format(
                nr=i + 1,
                sep=separator,
                td=td,
                inttime=inttime
            )

            stri2 += "{tpace:0>7}{sep}{tspm:0>4.1f}{sep}{thr:3.1f}".format(
                tpace=tpace,
                sep=separator,
                tspm=tspm,
                thr=thr
            )

            stri2 += "{sep}{tmaxhr:3.1f}{sep}{tdps:0>4.1f}".format(
                sep=separator,
                tmaxhr=tmaxhr,
                tdps=tdps
            )

            stri2 += "\n"

        tottime = "{totmin:0>2}:{totsec:0>2}".format(
            totmin=totmin + 60 * tothour,
            totsec=totsec)

        totaltime = tothour * 3600 + totmin * 60 + totsec

        avgspeed = totaldistance / totaltime
        avgpace = 500. / avgspeed
        avghr = tottimehr / totaltime
        avgsr = tottimespm / totaltime

        min = int(avgpace / 60)
        sec = int(10 * (avgpace - min * 60.)) / 10.
        pacestring = str(min) + ":" + str(sec)

        avgdps = totaldistance / (1.0 * totalstrokes)
        if isnan(avgdps):
            avgdps = 0

        stri1 = summarystring(totaldistance, totaltime, avgpace, avgsr,
                              avghr, maxhr, avgdps, 0,
                              readFile=self.readFile,
                              separator=separator)

        # print(stri1+stri2)

        copytocb(stri1 + stri2)

        return stri1 + stri2

    def shortstats(self):
        avghr = self.workdata['Avg HR'].mean()
        avgsr = self.workdata['Avg SR'].mean()
        maxhr = self.workdata['Max HR'].mean()
        maxsr = self.workdata['Max SR'].mean()
        totaldistance = self.workdata['Distance (m)'].sum()
        avgspeed = self.workdata['Avg Speed (m/s)'].mean()
        avgpace = 500 / avgspeed

        min = int(avgpace / 60)
        sec = int(10 * (avgpace - min * 60.)) / 10.
        pacestring = str(min) + ":" + str(sec)

        nr_rows = self.workdata.shape[0]

        totmin = 0
        totsec = 0

        for i in range(nr_rows):
            inttime = self.workdata['Time'].iloc[i]
            try:
                t = time.strptime(inttime, "%H:%M:%S")
            except ValueError:
                t = time.strptime(inttime, "%M:%S")

            totmin = totmin + t.tm_min
            totsec = totsec + t.tm_sec
            if (totsec > 60):
                totsec = totsec - 60
                totmin = totmin + 1

        stri = "=========WORK DATA=================\n"
        stri = stri + "Total Time     : " + \
            str(totmin) + ":" + str(totsec) + "\n"
        stri = stri + "Total Distance : " + str(totaldistance) + " m\n"
        stri = stri + "Average Pace   : " + pacestring + "\n"
        stri = stri + "Average HR     : " + str(int(avghr)) + " Beats/min\n"
        stri = stri + "Average SPM    : " + \
            str(int(10 * avgsr) / 10.) + " /min\n"
        stri = stri + "Max HR         : " + str(int(maxhr)) + " Beats/min\n"
        stri = stri + "Max SPM        : " + \
            str(int(10 * maxsr) / 10.) + " /min\n"
        stri = stri + "==================================="

        copytocb(stri)

        print(stri)


ftppowerperc = [55, 75, 90, 105, 120]
ftppowernames = ['UT3', 'UT2', 'UT1', 'AT', 'TR', 'AN']
hrzonenames = ['Rest','UT2','UT1','AT','TR','AN','max']

class rower:
    """ This class contains all the personal data about the rower

    * HR threshold values

    * C2 logbook username and password

    * weight category

    """

    def __init__(self, hrut2=142, hrut1=146, hrat=160,
                 hrtr=167, hran=180, hrmax=192,
                 c2username="",
                 c2password="",
                 weightcategory="hwt",
                 mc=72.5,
                 strokelength=1.35, ftp=226,
                 powerperc=ftppowerperc,
                 powerzones=ftppowernames,
                 hrzones=hrzonenames):
        self.ut2 = hrut2
        self.ut1 = hrut1
        self.at = hrat
        self.tr = hrtr
        self.an = hran
        self.max = hrmax
        self.c2username = c2username
        self.c2password = c2password
        self.ftp = ftp
        self.powerperc = powerperc
        self.powerzones = powerzones
        self.hrzones = hrzones
        if (weknowphysics == 1):
            self.rc = rowingphysics.crew(mc=mc, strokelength=strokelength)
        else:
            self.rc = 0
        if (weightcategory != "hwt") and (weightcategory != "lwt"):
            print("Weightcategory unrecognized. Set to hwt")
            weightcategory = "hwt"

        self.weightcategory = weightcategory

    def write(self, fileName):
        res = write_obj(self, fileName)


def roweredit(fileName="defaultrower.txt"):
    """ Easy editing or creation of a rower file.
    Mainly for using from the windows command line

    """

    try:
        r = pickle.load(open(fileName,'rb'))
    except IOError:
        print("Roweredit: File does not exist. Reverting to defaultrower.txt")
        r = getrower()
    except ImportError:
        print("Roweredit: File is not valid. Reverting to defaultrower.txt")
        r = getrower()

    try:
        rc = r.rc
    except AttributeError:
        if (weknowphysics == 1):
            rc = rowingphysics.crew(mc=70.0)
        else:
            rc = 0

    try:
        ftp = r.ftp
    except AttributeError:
        ftp = 225

    print("Functional Threshold Power")
    print(("Your Functional Threshold Power is set to {ftp}".format(
        ftp=ftp
    )))
    strin = input(
        'Enter new FTP (just ENTER to keep {ftp}:'.format(ftp=ftp))
    if (strin != ""):
        try:
            r.ftp = int(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    print("Heart Rate Training Bands")
    # hrmax
    print(("Your HR max is set to {hrmax} bpm".format(
        hrmax=r.max
    )))
    strin = input(
        'Enter HR max (just ENTER to keep {hrmax}):'.format(hrmax=r.max))
    if (strin != ""):
        try:
            r.max = int(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    # hrut2, hrut1
    print(("UT2 zone is between {hrut2} and {hrut1} bpm ({percut2:2.0f}-{percut1:2.0f}% of max HR)".format(
        hrut2=r.ut2,
        hrut1=r.ut1,
        percut2=100. * r.ut2 / r.max,
        percut1=100. * r.ut1 / r.max
    )))
    strin = input(
        'Enter UT2 band lower value (ENTER to keep {hrut2}):'.format(hrut2=r.ut2))
    if (strin != ""):
        try:
            r.ut2 = int(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    strin = input(
        'Enter UT2 band upper value (ENTER to keep {hrut1}):'.format(hrut1=r.ut1))
    if (strin != ""):
        try:
            r.ut1 = int(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    print(("UT1 zone is between {val1} and {val2} bpm ({perc1:2.0f}-{perc2:2.0f}% of max HR)".format(
        val1=r.ut1,
        val2=r.at,
        perc1=100. * r.ut1 / r.max,
        perc2=100. * r.at / r.max
    )))

    strin = input(
        'Enter UT1 band upper value (ENTER to keep {hrat}):'.format(hrat=r.at))
    if (strin != ""):
        try:
            r.at = int(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    print(("AT zone is between {val1} and {val2} bpm ({perc1:2.0f}-{perc2:2.0f}% of max HR)".format(
        val1=r.at,
        val2=r.tr,
        perc1=100. * r.at / r.max,
        perc2=100. * r.tr / r.max
    )))

    strin = input(
        'Enter AT band upper value (ENTER to keep {hrtr}):'.format(hrtr=r.tr))
    if (strin != ""):
        try:
            r.tr = int(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    print(("TR zone is between {val1} and {val2} bpm ({perc1:2.0f}-{perc2:2.0f}% of max HR)".format(
        val1=r.tr,
        val2=r.an,
        perc1=100. * r.tr / r.max,
        perc2=100. * r.an / r.max
    )))

    strin = input(
        'Enter TR band upper value (ENTER to keep {hran}):'.format(hran=r.an))
    if (strin != ""):
        try:
            r.an = int(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    print("")

    # weightcategory
    print(("Your weight category is set to {weightcategory}.".format(
        weightcategory=r.weightcategory
    )))
    strin = input(
        'Enter lwt for Light Weight, hwt for Heavy Weight, or just ENTER: ')
    if (strin != ""):
        if (strin == 'lwt'):
            r.weightcategory = strin
            print(("Setting to " + strin))
        elif (strin == 'hwt'):
            r.weightcategory = strin
            print(("Setting to " + strin))
        else:
            print("Value not recognized")

    print("")

    mc = rc.mc
    # weight
    strin = input("Enter weight in kg (or ENTER to keep {mc} kg):".format(
        mc=mc
    ))
    if (strin != ""):
        rc.mc = float(strin)

    # strokelength
    strin = input("Enter strokelength in m (or ENTER to keep {l} m:".format(
        l=rc.strokelength
    ))
    if (strin != ""):
        rc.strokelength = float(strin)

    r.rc = rc

    # c2username
    if (r.c2username != ""):
        print(("Your Concept2 username is set to {c2username}.".format(
            c2username=r.c2username
        )))
        strin = input('Enter new username (or just ENTER to keep): ')
        if (strin != ""):
            r.c2username = strin

    # c2password
    if (r.c2username == ""):
        print("We don't know your Concept2 username")
        strin = input('Enter new username (or ENTER to skip): ')
        r.c2username = strin

    if (r.c2username != ""):
        if (r.c2password != ""):
            print("We have your Concept2 password.")
            changeyesno = input(
                'Do you want to change/erase your password (y/n)')
            if changeyesno == "y":
                strin1 = getpass.getpass(
                    'Enter new password (or ENTER to erase):')
                if (strin1 != ""):
                    strin2 = getpass.getpass('Repeat password:')
                    if (strin1 == strin2):
                        r.c2password = strin1
                    else:
                        print("Error. Not the same.")
                if (strin1 == ""):
                    print("Forgetting your password")
                    r.c2password = ""
        elif (r.c2password == ""):
            print("We don't have your Concept2 password yet.")
            strin1 = getpass.getpass('Concept2 password (or ENTER to skip):')
            if (strin1 != ""):
                strin2 = getpass.getpass('Repeat password:')
                if (strin1 == strin2):
                    r.c2password = strin1
                else:
                    print("Error. Not the same.")

    r.write(fileName)

    print("Done")
    return 1

def boatedit(fileName="my1x.txt"):
    """ Easy editing or creation of a boat rigging data file.
    Mainly for using from the windows command line

    """

    try:
        rg = pickle.load(open(fileName,'rb'))
    except IOError:
        print("Boatedit: File does not exist. Reverting to my1x.txt")
        rg = getrigging()
    except (ImportError, ValueError):
        print("Boatedit: File is not valid. Reverting to my1x.txt")
        rg = getrigging()

    print("Number of rowers")
    # Lin
    print(("Your boat has {Nrowers} seats.".format(
        Nrowers=rg.Nrowers
    )))
    strin = input('Enter number of seats (just ENTER to keep {Nrowers}):'.format(
        Nrowers=rg.Nrowers
    ))
    if (strin != ""):
        try:
            rg.Nrowers = int(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    print("Rowing or sculling")
    # roworscull
    strin = input('Row (r) or scull (s) - ENTER to keep {roworscull}:'.format(
        roworscull=rg.roworscull
    ))
    if (strin == "s"):
        rg.roworscull = 'scull'
    elif (strin == "r"):
        rg.roworscull = 'row'

    print("Boat weight")
    # mb
    print(("Your {Nrowers} boat weighs {mb} kg".format(
        Nrowers=rg.Nrowers,
        mb=rg.mb
    )))
    strin = input('Enter boat weight including cox (just ENTER to keep {mb}):'.format(
        mb=rg.mb
    ))
    if (strin != ""):
        try:
            rg.mb = float(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    print("Rigging Data")
    # Lin
    print(("Your inboard is set to {lin} m".format(
        lin=rg.lin
    )))
    strin = input('Enter inboard (just ENTER to keep {lin} m):'.format(
        lin=rg.lin
    ))
    if (strin != ""):
        try:
            rg.lin = float(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    print(("Your scull/oar length is set to {lscull} m".format(
        lscull=rg.lscull
    )))
    print("For this number, you need to subtract half of the blade length from the classical oar/scull length measurement")
    strin = input('Enter length (subtract half of blade length, just ENTER to keep {lscull}):'.format(
        lscull=rg.lscull
    ))
    if (strin != ""):
        try:
            rg.lscull = float(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    if (rg.roworscull == 'row'):
        print(("Your spread is set to {spread} m".format(
            spread=rg.spread
        )))
        strin = input('Enter new spread (or ENTER to keep {spread} m):'.format(
            spread=rg.spread
        ))
        if (strin != ""):
            try:
                rg.spread = float(spread)
            except ValueError:
                print("Not a valid number. Keeping original value")
    else:
        print(("Your span is set to {span} m".format(
            span=rg.span
        )))
        strin = input('Enter new span (or ENTER to keep {span} m):'.format(
            span=rg.span
        ))
        if (strin != ""):
            try:
                rg.span = float(span)
            except ValueError:
                print("Not a valid number. Keeping original value")

    # Blade Area
    print(("Your blade area is set to {bladearea} m2 (total blade area per rower, take two blades for scullers)".format(
        bladearea=rg.bladearea
    )))
    strin = input('Enter blade area (just ENTER to keep {bladearea} m2):'.format(
        bladearea=rg.bladearea
    ))
    if (strin != ""):
        try:
            rg.bladearea = float(strin)
        except ValueError:
            print("Not a valid number. Keeping original value")

    # Catch angle
    catchangledeg = -np.degrees(rg.catchangle)

    print("We define catch angle as follows.")
    print(" - 0 degrees is a catch with oar shaft perpendicular to the boat")
    print(" - 90 degrees is a catch with oar shaft parallel to the boat")
    print(" - Use positive values for normal catch angles")
    print("Your catch angle is {catchangledeg} degrees.")
    strin = input('Enter catch angle in degrees (or ENTER to keep {catchangledeg}):'.format(
        catchangledeg=catchangledeg
    ))
    if (strin != ""):
        try:
            rg.catchangle = -np.radians(float(strin))
        except ValueError:
            print("Not a valid number. Keeping original value")

    write_obj(rg, fileName)

    print("Done")
    return 1

def addpowerzones(df, ftp, powerperc):
    number_of_rows = df.shape[0]

    df['pw_ut2'] = np.zeros(number_of_rows)
    df['pw_ut1'] = np.zeros(number_of_rows)
    df['pw_at'] = np.zeros(number_of_rows)
    df['pw_tr'] = np.zeros(number_of_rows)
    df['pw_an'] = np.zeros(number_of_rows)
    df['pw_max'] = np.zeros(number_of_rows)

    percut2, percut1, percat, perctr, percan = np.array(powerperc) / 100.

    ut2, ut1, at, tr, an = ftp * np.array(powerperc) / 100.

    df['limpw_ut2'] = percut2 * ftp
    df['limpw_ut1'] = percut1 * ftp
    df['limpw_at'] = percat * ftp
    df['limpw_tr'] = perctr * ftp
    df['limpw_an'] = percan * ftp

    # create the columns containing the data for the colored bar chart
    # attempt to do this in a way that doesn't generate dubious copy warnings
    try:
        mask = (df[' Power (watts)'] <= ut2) & (
            df[' Stroke500mPace (sec/500m)'] < 360)
        df.loc[mask, 'pw_ut2'] = df.loc[mask, ' Power (watts)']

        mask = (df[' Power (watts)'] <= ut1) & (df[' Power (watts)']
                                                >= ut2) & (df[' Stroke500mPace (sec/500m)'] < 360)
        df.loc[mask, 'pw_ut1'] = df.loc[mask, ' Power (watts)']

        mask = (df[' Power (watts)'] <= at) & (df[' Power (watts)']
                                               >= ut1) & (df[' Stroke500mPace (sec/500m)'] < 360)
        df.loc[mask, 'pw_at'] = df.loc[mask, ' Power (watts)']

        mask = (df[' Power (watts)'] <= tr) & (df[' Power (watts)']
                                               >= at) & (df[' Stroke500mPace (sec/500m)'] < 360)
        df.loc[mask, 'pw_tr'] = df.loc[mask, ' Power (watts)']

        mask = (df[' Power (watts)'] <= an) & (df[' Power (watts)']
                                               >= tr) & (df[' Stroke500mPace (sec/500m)'] < 360)
        df.loc[mask, 'pw_an'] = df.loc[mask, ' Power (watts)']

        mask = (df[' Power (watts)'] >= an) & (
            df[' Stroke500mPace (sec/500m)'] < 360)
        df.loc[mask, 'pw_max'] = df.loc[mask, ' Power (watts)']
    except TypeError:
        pass

    df = df.fillna(method='ffill')

    return df

def addzones(df, ut2, ut1, at, tr, an, mmax):
        # define an additional data frame that will hold the multiple bar plot data and the hr
        # limit data for the plot, it also holds a cumulative distance column

    number_of_rows = df.shape[0]

    df['hr_ut2'] = np.zeros(number_of_rows)
    df['hr_ut1'] = np.zeros(number_of_rows)
    df['hr_at'] = np.zeros(number_of_rows)
    df['hr_tr'] = np.zeros(number_of_rows)
    df['hr_an'] = np.zeros(number_of_rows)
    df['hr_max'] = np.zeros(number_of_rows)

    df['lim_ut2'] = ut2
    df['lim_ut1'] = ut1
    df['lim_at'] = at
    df['lim_tr'] = tr
    df['lim_an'] = an
    df['lim_max'] = mmax

    # create the columns containing the data for the colored bar chart
    # attempt to do this in a way that doesn't generate dubious copy warnings
    mask = (df[' HRCur (bpm)'] <= ut2) & (
        df[' Stroke500mPace (sec/500m)'] < 360)
    df.loc[mask, 'hr_ut2'] = df.loc[mask, ' HRCur (bpm)']

    mask = (df[' HRCur (bpm)'] <= ut1) & (df[' HRCur (bpm)']
                                          >= ut2) & (df[' Stroke500mPace (sec/500m)'] < 360)
    df.loc[mask, 'hr_ut1'] = df.loc[mask, ' HRCur (bpm)']

    mask = (df[' HRCur (bpm)'] <= at) & (df[' HRCur (bpm)']
                                         >= ut1) & (df[' Stroke500mPace (sec/500m)'] < 360)
    df.loc[mask, 'hr_at'] = df.loc[mask, ' HRCur (bpm)']

    mask = (df[' HRCur (bpm)'] <= tr) & (df[' HRCur (bpm)']
                                         >= at) & (df[' Stroke500mPace (sec/500m)'] < 360)
    df.loc[mask, 'hr_tr'] = df.loc[mask, ' HRCur (bpm)']

    mask = (df[' HRCur (bpm)'] <= an) & (df[' HRCur (bpm)']
                                         >= tr) & (df[' Stroke500mPace (sec/500m)'] < 360)
    df.loc[mask, 'hr_an'] = df.loc[mask, ' HRCur (bpm)']

    mask = (df[' HRCur (bpm)'] >= an) & (df[' Stroke500mPace (sec/500m)'] < 360)
    df.loc[mask, 'hr_max'] = df.loc[mask, ' HRCur (bpm)']

    # fill cumulative distance column with cumulative distance
    # ignoring resets to lower distance values
    try:
        cumdist = df['cum_dist']
    except KeyError:
        df['cum_dist'] = np.zeros(number_of_rows)

        df['cum_dist'] = make_cumvalues(df[' Horizontal (meters)'])[0]

    df = df.fillna(method='ffill')

    return df

Nspm = 30
Nvw = 40
Nvb = 550

def getaddress(spm,vw,vb):
    spmmin = 15
    spmmax = 45

    spmrel = (spm-spmmin)/float(spmmax-spmmin)
    spmrel = min([max([spmrel,0]),1])

    i = int((Nspm-1)*spmrel)

    vwmin = -10
    vwmax = +10

    vwrel = (vw-vwmin)/float(vwmax-vwmin)
    vwrel = min([max([vwrel,0]),1])

    j = int((Nvw-1)*vwrel)

    vbmin = 2.5
    vbmax = 8.0

    vbrel = (vb-vbmin)/float(vbmax-vbmin)
    vbrel = min([max([vbrel,0]),1])


    k = int((Nvb-1)*vbrel)

    return i,j,k

class rowingdata:
    """ This is the main class. Read the data from the csv file and do all
    kinds
    of cool stuff with it.

    Usage: row=rowingdata.rowingdata(csvfile="testdata.csv",
                                       rowtype="Indoor Rower",
                                       absolutetimestamps=False,
                                       rower=rr,
                                       )


    If absolutetimestamps is set to True, the time stamp info in the
    main dataframe will be seconds since 1-1-1970. The default is
    seconds since workout start.

    The default rower looks for a defaultrower.txt file. If it is not found,
    it reverts to some arbitrary rower.


    """

    def __init__(self, *args, **kwargs):

        if 'debug' in kwargs:
            debug = kwargs['debug']
        else:
            debug = False

        self.debug = debug

        if 'csvfile' in kwargs:
            readFile = kwargs['csvfile']
        else:
            readFile = None

        if 'absolutetimestamps' in kwargs:
            self.absolutetimestamps = kwargs['absolutetimestamps']
        else:
            self.absolutetimestamps = False

        if args:
            readFile = args[0]
            warnings.warn(
                "Depreciated. Use rowingdata(csvfile=csvfile)", UserWarning)

        rwr = kwargs.get('rower', rower())

        rowtype = kwargs.get('rowtype', 'Indoor Rower')

        sled_df = DataFrame()
        if 'df' in kwargs:
            sled_df = kwargs['df']
            # new_index=range(len(sled_df))
            # sled_df=sled_df.reindex(index=new_index)
            readFile = 0
        elif readFile:
            try:
                try:
                    sled_df = pd.read_csv(readFile,encoding='utf-8')
                except IOError:
                    sled_df = pd.read_csv(readFile + '.gz',encoding='utf-8')
            except IOError:
                try:
                    f = open(readFile)
                    sled_df = pd.read_csv(f)
                    f.close()
                except IOError:
                    try:
                        f = open(readFile + '.gz')
                        sled_df = pd.read_csv(f)
                        f.close()
                    except:
                        sled_df = pd.DataFrame()
            except UnicodeEncodeError:
                try:
                    f = open(readFile)
                    sled_df = pd.read_csv(f)
                    f.close()
                except IOError:
                    try:
                        f = open(readFile + '.gz')
                        sled_df = pd.read_csv(f)
                        f.close()
                    except:
                        sled_df = pd.DataFrame()



        if readFile:
            try:
                self.readfilename = readFile.name
            except AttributeError:
                self.readfilename = readFile
        else:
            self.readfilename = 'rowing dataframe'

        self.readFile = readFile
        self.rwr = rwr
        self.rowtype = rowtype

        self.empty = False
        if sled_df.empty:
            self.empty = True

        othernames = ['catch','finish','peakforceangle',
                      'wash','slip','index',
                      'cum_dist','hr_an','hr_at','hr_tr','hr_ut1','hr_ut2',
                      'lim_an','lim_at','lim_tr','lim_ut1','lim_ut2',
                      'limpw_an','limpw_at','limpw_tr',
                      'limpw_ut1','limpw_ut2',
                      'pw_an','pw_at','pw_max','pw_tr','pw_ut1','pw_ut2',
                      'lim_max','hr_max',
                      ' latitude',' longitude']

        # check for missing column names
        mandatorynames = [
            'TimeStamp (sec)',
            ' Horizontal (meters)',
            ' Cadence (stokes/min)',
            ' HRCur (bpm)',
            ' Stroke500mPace (sec/500m)',
            ' Power (watts)',
            ' DriveLength (meters)',
            ' StrokeDistance (meters)',
            ' DriveTime (ms)',
            ' DragFactor',
            ' StrokeRecoveryTime (ms)',
            ' AverageDriveForce (lbs)',
            ' AverageBoatSpeed (m/s)',
            ' PeakDriveForce (lbs)',
            ' AverageDriveForce (N)',
            ' PeakDriveForce (N)',
            ' lapIdx',
            ' ElapsedTime (sec)',
            ' Calories (kCal)',
            ' WorkoutState',
        ]

        self.defaultnames = othernames+mandatorynames

        if ' ElapsedTime (sec)' not in sled_df.columns and not sled_df.empty:
            sled_df[' ElapsedTime (sec)'] = sled_df['TimeStamp (sec)']-sled_df.loc[0,'TimeStamp (sec)']

        for name in mandatorynames:
            if name not in sled_df.columns and not sled_df.empty:
                if debug:
                    print(name + ' is not found in file')
                sled_df[name] = 0
                sled_df.index = list(range(len(sled_df.index)))
                if name == 'TimeStamp (sec)':
                    time = sled_df['TimeStamp (sec utc)']
                    sled_df[name] = time
                if name == ' ElapsedTime (sec)':
                    elapsedtime = sled_df['TimeStamp (sec)'] - \
                        sled_df.loc[0, 'TimeStamp (sec)']
                    sled_df[name] = elapsedtime
                if name == ' WorkoutState':
                    sled_df[name] = 4
                if name == ' Calories (kCal)':
                    sled_df[name] = 1
                if name == ' Stroke500mPace (sec/500m)':
                    dd = sled_df[' Horizontal (meters)'].diff()
                    dt = sled_df[' ElapsedTime (sec)'].diff()
                    velo = dd / dt
                    sled_df[name] = 500. / velo
                if name == ' AverageBoatSpeed (m/s)':
                    try:
                        velo = 500./sled_df[' Stroke500mPace (sec/500m)']
                    except (KeyError,ValueError):
                        dd = sled_df[' Horizontal (meters)'].diff()
                        dt = sled_df[' ElapsedTime (sec)'].diff()
                        velo = dd / dt

                    sled_df[name] = velo
                if name == ' AverageDriveForce (lbs)':
                    try:
                        forcen = sled_df[' AverageDriveForce (N)']
                        sled_df[name] = forcen / lbstoN
                    except KeyError:
                        pass
                if name == ' AverageDriveForce (N)':
                    try:
                        forcelbs = sled_df[' AverageDriveForce (lbs)']
                        sled_df[name] = forcelbs * lbstoN
                    except KeyError:
                        pass
                if name == ' PeakDriveForce (N)':
                    try:
                        forcelbs = sled_df[' PeakDriveForce (lbs)']
                        sled_df[name] = forcelbs * lbstoN
                    except KeyError:
                        pass
                if name == ' PeakDriveForce (lbs)':
                    try:
                        forcen = sled_df[' PeakDriveForce (N)']
                        sled_df[name] = forcen / lbstoN
                    except KeyError:
                        pass
                if name == ' Cadence (stokes/min)':
                    try:
                        spm = sled_df[' Cadence (strokes/min)']
                        if debug:
                            print('Cadence found')
                        sled_df[name] = spm
                    except KeyError:
                        pass

        mandatorynames.remove(' lapIdx')

        try:
            sled_df[mandatorynames] = sled_df[mandatorynames].apply(pd.to_numeric,errors='coerce',axis=1)
        except KeyError:
            pass

        if len(sled_df):
            # Remove zeros from HR
            hrmean = sled_df[' HRCur (bpm)'].mean()
            hrstd = sled_df[' HRCur (bpm)'].std()

            if hrmean != 0 and hrstd != 0:
                sled_df[' HRCur (bpm)'].replace(to_replace=0,
                                                method='ffill',
                                                inplace=True)

            self.dragfactor = sled_df[' DragFactor'].mean()
            # do stroke count
            dt = sled_df['TimeStamp (sec)'].diff()
            dstroke = dt*sled_df[' Cadence (stokes/min)']/60.
            self.stroke_count = int(dstroke.sum())
        else:
            self.dragfactor = 0
            self.stroke_count = 0

        # get the date of the row
        starttime = 0
        if not sled_df.empty:
            try:
                starttime = sled_df['TimeStamp (sec)'].values[0]
            except KeyError as IndexError:
                starttime = 0

        # create start time timezone aware time object
        try:
            self.rowdatetime = arrow.get(starttime).datetime
        except ValueError:
            self.rowdatetime = datetime.datetime.utcnow()

        # remove the start time from the time stamps
        if not self.absolutetimestamps and len(sled_df):
            sled_df['TimeStamp (sec)'] = sled_df['TimeStamp (sec)'] - \
                sled_df['TimeStamp (sec)'].values[0]

        number_of_columns = sled_df.shape[1]
        number_of_rows = sled_df.shape[0]

        if not sled_df.empty:
            try:
                dt = sled_df['TimeStamp (sec)'].diff()
                try:
                    dt.iloc[0] = dt.iloc[1] # replaced ix with iloc
                except:
                    dt.loc[dt.index[0]] = dt.loc[dt.index[0]]

                dt.fillna(inplace=True, method='ffill')
                dt.fillna(inplace=True, method='bfill')
                strokenumbers = pd.Series(
                    np.cumsum(dt*sled_df[' Cadence (stokes/min)']/60.)
                )
                if strokenumbers.isnull().all():
                    strokenumbers.loc[:] = 0
                else:
                    strokenumbers.fillna(inplace=True, method='ffill')
                    strokenumbers.fillna(inplace=True, method='bfill')

                sled_df[' Stroke Number'] = strokenumbers.astype('int')
            except KeyError:
                if debug:
                    print("Could not calculate stroke number")
                else:
                    pass

        # add driveenergy
        try:
            sled_df['driveenergy'] = 60.*sled_df[' Power (watts)']/sled_df[' Cadence (stokes/min)']
        except KeyError:
            sled_df['driveenergy'] = 0

        # these parameters are handy to have available in other routines
        self.number_of_rows = number_of_rows

        # add HR zone data to dataframe
        if len(sled_df):
            self.df = addzones(sled_df, self.rwr.ut2,
                               self.rwr.ut1,
                               self.rwr.at,
                               self.rwr.tr,
                               self.rwr.an,
                               self.rwr.max
            )
        else:
            self.df = sled_df

        if len(sled_df):
            # Remove "logging data" - not strokes
            self.df = self.df[self.df[' WorkoutState'] != 12]

            # Cadence to float
            self.df[' Cadence (stokes/min)'] = self.df[' Cadence (stokes/min)'].astype(float)

            self.df = addpowerzones(self.df, self.rwr.ftp, self.rwr.powerperc)
        self.index = self.df.index

        # duration
        if len(sled_df):
            self.duration = self.df['TimeStamp (sec)'].max()-self.df['TimeStamp (sec)'].min()
        else:
            self.duration = 0


    def __add__(self, other):
        self_df = self.df.copy()
        other_df = other.df.copy()

        if self.empty:
            return other

        if other.empty:
            return self

        if not self.absolutetimestamps:
            # starttimeunix=time.mktime(self.rowdatetime.utctimetuple())
            starttimeunix1 = arrow.get(self.rowdatetime).timestamp()
            self_df['TimeStamp (sec)'] = self_df['TimeStamp (sec)'] + \
                starttimeunix1
        if not other.absolutetimestamps:
            # starttimeunix=time.mktime(other.rowdatetime.utctimetuple())
            starttimeunix2 = arrow.get(other.rowdatetime).timestamp()
            other_df['TimeStamp (sec)'] = other_df['TimeStamp (sec)'] + \
                starttimeunix2

        # determine overlap
        overlap1 = self_df['TimeStamp (sec)'].max() > starttimeunix2 and starttimeunix1 < starttimeunix2
        overlap2 = other_df['TimeStamp (sec)'].max() > starttimeunix1 and starttimeunix2 < starttimeunix1

        # remove overlap
        if overlap1:
            delta = self_df['TimeStamp (sec)'].max() - starttimeunix2
            if delta < 60:
                starttimeunix2 += delta+0.1
                other_df['TimeStamp (sec)'] += delta+0.1

        if overlap2:
            delta = other_df['TimeStamp (sec)'].max() - starttimeunix1
            if delta < 60:
                starttimeunix1 += delta+0.1
                self_df['TimeStamp (sec)'] += delta+0.1

        lapids = self_df[' lapIdx'].unique()
        otherlapids = other_df[' lapIdx'].unique()
        overlapping = list(set(lapids) & set(otherlapids))
        while overlapping:
            try:
                other_df[' lapIdx'] = other_df[' lapIdx'].apply(
                    lambda n: n + 1)
            except TypeError:
                other_df[' lapIdx'] = other_df[' lapIdx'].apply(
                    lambda s: 'i' + s)
            otherlapids = other_df[' lapIdx'].unique()
            overlapping = list(set(lapids) & set(otherlapids))

        self_df = pd.merge(self_df, other_df, how='outer')
        # drop duplicates
        self_df.drop_duplicates(subset='TimeStamp (sec)',
                                keep='first', inplace=True)
        self_df = self_df.sort_values(by='TimeStamp (sec)', ascending=1)
        self_df = self_df.fillna(method='ffill')
        self_df.reset_index(drop=True,inplace=True)

        # recalc cum_dist
        # this needs improvement. If Elapsed Distance is measured
        # inconsistently across the two dataframes, it leads to errors.
        self_df['cum_dist'] = make_cumvalues(
            self_df[' Horizontal (meters)'])[0]
        # self_df.to_csv('C:/Downloads/debug.csv')
        return rowingdata(df=self_df, rower=self.rwr,
                          rowtype=self.rowtype,
                          absolutetimestamps=self.absolutetimestamps)

    def change_drag(self, dragfactor):
        self.df[' DragFactor'] = dragfactor
        self.dragfactor = dragfactor

    def getvalues(self, keystring):
        """ Just a tool to get a column of the row data as a numpy array

        You can also just access row.df[keystring] to get a pandas Series

        """

        if self.empty:
            return np.array([])

        return self.df[keystring].values

    def __len__(self):
        return len(self.df)

    def get_additional_metrics(self):
        cols = self.df.columns.values
        dif = np.setdiff1d(cols,self.defaultnames)

        additionalmetrics = []

        for c in dif:
            try:
                test = self.df[c].apply(lambda x:float(x))
                additionalmetrics.append(c)
            except ValueError:
                pass
            except TypeError:
                pass

        return additionalmetrics


    def check_consistency(self, threshold=20, velovariation=1.e-4):
        data = self.df

        result = {}

        if self.empty:
            result['velo_time_distance'] = True
            result['velo_valid'] = True
            return result

        # velocity integrated over time must equal total distance
        velo = 500. / data[' Stroke500mPace (sec/500m)']

        # clip extreme values
        velo = velo.clip(upper=10.0)

        rowtime = data['TimeStamp (sec)']
        totaldfromvelo = integrate.trapz(velo, x=rowtime)
        totaldfromcumdist = data['cum_dist'].max()

        r1 = (100. - threshold) / 100.
        r2 = (100. + threshold) / 100.

        if np.isfinite(totaldfromvelo):
            testresult = totaldfromvelo * r1 <= totaldfromcumdist <= totaldfromvelo * r2
        else:
            testresult = True

        result['velo_time_distance'] = testresult

        # standard deviation of velocity must be non-zero
        try:
            result['velo_valid'] = (velo.std() / velo.mean() >= velovariation)
        except ZeroDivisionError:
            result['velo_valid'] = True

        return result

    def repair(self):
        data = self.df

        checks = self.check_consistency()

        if not checks['velo_time_distance'] and checks['velo_valid']:
            # calculate distance from velocity and time
            velo = 500. / data[' Stroke500mPace (sec/500m)']
            time = data['TimeStamp (sec)']
            dt = np.nan_to_num(time.diff())
            distance = np.cumsum(dt * velo)
            data['cum_dist'] = distance
            self.df = data
        elif not checks['velo_valid']:
            time = data['TimeStamp (sec)']
            distance = data[' Horizontal (meters)']
            dt = np.nan_to_num(time.diff())
            dx = np.nan_to_num(distance.diff())
            pace = 500. * dt / dx
            data[' Stroke500mPace (sec/500m)'] = pace

    def write_csv(self, writeFile, gzip=False):
        data = self.df
        data = data.drop(['index',
                          'hr_ut2',
                          'hr_ut1',
                          'hr_at',
                          'hr_tr',
                          'hr_an',
                          'hr_max',
                          'lim_ut2',
                          'lim_ut1',
                          'lim_at',
                          'lim_tr',
                          'lim_an',
                          'lim_max',
                          'pw_ut2',
                          'pw_ut1',
                          'pw_at',
                          'pw_tr',
                          'pw_an',
                          'pw_max',
                          'limpw_ut2',
                          'limpw_ut1',
                          'limpw_at',
                          'limpw_tr',
                          'limpw_an',
                          'limpw_max',
                          ], 1, errors='ignore')

        # add time stamp to
        if not self.absolutetimestamps and not self.empty:
            try:
                # starttimeunix=time.mktime(self.rowdatetime.utctimetuple())
                starttimeunix = arrow.get(self.rowdatetime).timestamp()
            except:
                starttimeunix = time.mktime(
                    datetime.datetime.now().utctimetuple())
            data['TimeStamp (sec)'] = data['TimeStamp (sec)'] + starttimeunix

        if gzip:
            return data.to_csv(writeFile + '.gz', index_label='index',
                               compression='gzip')
        else:
            return data.to_csv(writeFile, index_label='index')

    def use_impellerdata(self):
        df = self.df
        try:
            df[' AverageBoatSpeed (m/s)'] = df['ImpellerSpeed']
            df[' Horizontal (meters)'] = df['ImpellerDistance']
            dp = pd.DataFrame({'x':df[' ElapsedTime (sec)'].values,'y':df[' Horizontal (meters)'].values})
            dd = dp.dropna(axis=0,how='any')['y'].diff()
            dt = dp.dropna(axis=0,how='any')['x'].diff()
            velo = dd / dt
            df[' Stroke500mPace (sec/500m)'] = 500. / df[' AverageBoatSpeed (m/s)']
        except KeyError:
            return False

        self.df = df

        return True

    def extend_data(self):
        df = self.df
        l = len(df)
        nr = 10
        if l<10:
            nr = l-1
        dlat = (df.loc[l-1,' latitude']-df.loc[l-nr,' latitude'])/float(nr-1)
        dlon = (df.loc[l-1,' longitude']-df.loc[l-nr,' longitude'])/float(nr-1)
        dt = (df.loc[l-1, ' ElapsedTime (sec)']-df.loc[l-nr,' ElapsedTime (sec)'])/float(nr-1)
        tnew = []
        latnew = []
        lonnew = []
        unixtnew = []
        tz = df.loc[l-1,' ElapsedTime (sec)']
        unixtz = df.loc[l-1,'TimeStamp (sec)']
        latz = df.loc[l-1,' latitude']
        lonz = df.loc[l-1,' longitude']
        for i in range(10):
            tz += dt
            unixtz += dt
            latz += dlat
            lonz += dlon
            tnew.append(tz)
            latnew.append(latz)
            lonnew.append(lonz)
            unixtnew.append(unixtz)

        df = df.append(pd.DataFrame({
            'TimeStamp (sec)':unixtnew,
            ' ElapsedTime (sec)':tnew,
            ' latitude': latnew,
            ' longitude': lonnew,
        }))

        df.interpolate(inplace=True)
        df = df.fillna(method='ffill',axis=1)
        df['index'] = range(len(df))
        df.set_index('index',inplace=True)

        self.df = df

    def calc_dist_from_gps(self):
        df = self.df
        df['gps_dx'] = 0*df[' latitude']
        for i in range(len(df)-1):
            lat1 = df.loc[i,' latitude']
            lat2 = df.loc[i+1,' latitude']
            lon1 = df.loc[i,' longitude']
            lon2 = df.loc[i+1,' longitude']
            df.loc[i+1,'gps_dx'] = 1000*geo_distance(lat1,lon1,lat2,lon2)[0]

        df['gps_dist_calculated'] = df['gps_dx'].cumsum()

    def use_gpsdata(self):
        df = self.df
        try:
            df[' AverageBoatSpeed (m/s)'] = df['GPSSpeed']
            df[' Horizontal (meters)'] = df['GPSDistance']
            dp = pd.DataFrame({'x':df[' ElapsedTime (sec)'].values,'y':df[' Horizontal (meters)'].values})
            dd = dp.dropna(axis=0,how='any')['y'].diff()
            dt = dp.dropna(axis=0,how='any')['x'].diff()
            velo = dd / dt
            df[' Stroke500mPace (sec/500m)'] = 500. / df[' AverageBoatSpeed (m/s)']
        except KeyError:
            return False

        self.df = df

        return True

    def get_instroke_columns(self):
        cols = []
        for c in self.df.columns:
            try:
                d = self.df[c].str[1:-1].str.split(',',expand=True)
                a = d[1]
                cols.append(c)
            except:
                pass

        return cols

    def add_instroke_maxminpos(self,c):
        df = self.get_instroke_data(c)
        aantalcol = len(df.columns)
        minpos = aantalcol/10
        dfnorm = df.copy().iloc[:,minpos:] # replaced ix with iloc
        min_idxs = dfnorm.idxmin(axis=1)
        max_idxs = dfnorm.idxmax(axis=1)

        minpos = min_idxs/float(aantalcol)
        maxpos = max_idxs/float(aantalcol)

        f = self.df['TimeStamp (sec)'].diff().mean()
        if f != 0 and not np.isnan(f):
            windowsize = 2* (int(10./(f)))+1
        else:
            windowsize = 1

        if windowsize > 3 and windowsize < len(min_idxs):
            minpos = savgol_filter(minpos,windowsize,1)
            maxpos = savgol_filter(maxpos,windowsize,1)

        self.df[c+'_minpos'] = minpos
        self.df[c+'_maxpos'] = maxpos

    def add_instroke_diff(self,c):
        df = self.get_instroke_data(c)
        dfnorm = df.copy()
        dfnorm = (dfnorm.transpose()/dfnorm.transpose().max()).transpose()


        aantalcol = len(dfnorm.columns)
        diff = dfnorm.diff(axis=0)**2

        diff_c = diff.transpose().sum()/float(aantalcol)
        f = self.df['TimeStamp (sec)'].diff().mean()
        if f != 0 and not np.isnan(f):
            windowsize = 2* (int(10./(f)))+1
        else:
            windowsize = 1

        if windowsize > 3 and windowsize < len(diff_c):
            diff_c = savgol_filter(diff_c,windowsize,1)

        self.df[c+'_diff'] = diff_c


    def add_instroke_metrics(self,c):
        df = self.get_instroke_data(c)
        dfnorm = df.copy().abs()
        dfnorm = (dfnorm.transpose()/dfnorm.transpose().max()).transpose()


        aantalcol = len(dfnorm.columns)
        markers = (np.arange(4))*aantalcol/4

        # replaced ix with iloc in below
        first = dfnorm.iloc[:,markers[0]:markers[1]].mean(axis=1).rolling(10,min_periods=1).std()
        second = dfnorm.iloc[:,markers[1]+1:markers[2]].mean(axis=1).rolling(10,min_periods=1).std()
        third = dfnorm.iloc[:,markers[2]+1:markers[3]].mean(axis=1).rolling(10,min_periods=1).std()
        fourth = dfnorm.iloc[:,markers[3]+1:].mean(axis=1).rolling(10,min_periods=1).std()

        self.df[c+'_q1'] = first
        self.df[c+'_q2'] = second
        self.df[c+'_q3'] = third
        self.df[c+'_q4'] = fourth

    def set_instroke_metrics(self):
        cols = self.get_instroke_columns()
        for c in cols:
            self.add_instroke_metrics(str(c))
            self.add_instroke_diff(str(c))
            self.add_instroke_maxminpos(str(c))

    def get_instroke_data(self,column_name):
        df = self.df[column_name].str[1:-1].str.split(',',expand=True)
        df = df.apply(pd.to_numeric, errors = 'coerce')

        return df

    def plot_instroke(self,column_name):
        df  = self.get_instroke_data(column_name)

        mean_vals = df.median()
        min_vals = df.quantile(q=0.05)
        max_vals = df.quantile(q=0.95)

        q25 = df.quantile(q = 0.25)
        q75 = df.quantile(q = 0.75)

        df_plot = DataFrame({
            'mean':mean_vals,
            'max':max_vals,
            'q75':q75,
            'q25':q25,
            'min':min_vals,
            })

        df_plot.plot()

        plt.show()

    def get_plot_instroke(self,column_name):
        df  = self.get_instroke_data(column_name)

        mean_vals = df.median()
        min_vals = df.quantile(q=0.05)
        max_vals = df.quantile(q=0.95)

        q25 = df.quantile(q = 0.25)
        q75 = df.quantile(q = 0.75)

        df_plot = DataFrame({
            'mean':mean_vals,
            'max':max_vals,
            'q75':q75,
            'q25':q25,
            'min':min_vals,
            })

        fig1 = figure.Figure(figsize=(12,10))
        ax = fig1.add_subplot(111)
        df_plot.plot(ax=ax)

        return fig1


    def spm_fromtimestamps(self):
        if not self.empty:
            df = self.df
            dt = (df[' DriveTime (ms)'] + df[' StrokeRecoveryTime (ms)']) / 1000.
            spm = 60. / dt
            df[' Cadence (stokes/min)'] = spm
            self.df = df

    def erg_recalculatepower(self):
        if not self.empty:
            df = self.df
            velo = df[' Speed (m/sec)']
            pwr = 2.8 * velo**3
            df[' Power (watts)'] = pwr
            self.df = df

    def exporttotcx(self, fileName, notes="Exported by Rowingdata",
                    sport="Other"):
        if not self.empty:
            df = self.df

            writetcx.write_tcx(
                fileName,
                df,
                row_date=self.rowdatetime.isoformat(), notes=notes,
                sport=sport
            )
        else:
            emptytcx = writetcx.get_empty_tcx()
            bytes = emptytcx.encode(encoding='UTF-8')
            with open(fileName,'wb+') as f_out:
                f_out.write(bytes)


    def exporttogpx(self, fileName, notes="Exported by Rowingdata"):
        if not self.empty:
            df = self.df
            gpxwrite.write_gpx(fileName,
                               df,
                               row_date=self.rowdatetime.isoformat(),
                               notes=notes)
        else:
            emptygpx = gpxwrite.empty_gpx
            bytes = emptygpx.encode(encoding='UTF-8')
            with open(fileName,'wb+') as f_out:
                f_out.write(bytes)

    def intervalstats(self, separator='|'):
        """ Used to create a nifty text summary, one row for each interval

        Also copies the string to the clipboard (handy!)

        Works for painsled (both iOS and desktop version) because they use
        the lapIdx column

        """

        if self.empty:
            return ""

        df = self.df
        df['deltat'] = df['TimeStamp (sec)'].diff()

        workoutstateswork = [1, 4, 5, 8, 9, 6, 7]
        workoutstatesrest = [3]
        workoutstatetransition = [0, 2, 10, 11, 12, 13]

        intervalnrs = pd.unique(df[' lapIdx'])

        stri = "Workout Details\n"
        stri += "#-{sep}SDist{sep}-Split-{sep}-SPace-{sep}-Pwr-{sep}SPM-{sep}AvgHR{sep}MaxHR{sep}DPS-\n".format(
            sep=separator
        )

        previousdist = 0.0
        # previoustime=0.0
        previoustime = df['TimeStamp (sec)'].min()

        for idx in intervalnrs:
            # td = df[df[' lapIdx'] == idx]
            mask = df[' lapIdx'] == idx
            td = df[mask]

            # assuming no stroke type info
            tdwork = td

            avghr = wavg(tdwork,' HRCur (bpm)','deltat')
            maxhr = tdwork[' HRCur (bpm)'].max()
            avgspm = wavg(tdwork,' Cadence (stokes/min)','deltat')
            avgpower = wavg(tdwork,' Power (watts)','deltat')

            intervaldistance = tdwork[' Horizontal (meters)'].max()

            previousdist = tdwork['cum_dist'].max()

            intervalduration = tdwork['TimeStamp (sec)'].max() - previoustime
            previoustime = tdwork['TimeStamp (sec)'].max()

            intervalpace = 500. * intervalduration / intervaldistance
            avgdps = intervaldistance / (intervalduration * avgspm / 60.)
            if isnan(avgdps) or isinf(avgdps):
                avgdps = 0

            stri += interval_string(idx + 1, intervaldistance, intervalduration,
                                    intervalpace, avgspm,
                                    avghr, maxhr, avgdps, avgpower,
                                    separator=separator)

        return stri

    def intervalstats_values(self,debug=False):
        """ Used to create a nifty text summary, one row for each interval

        Also copies the string to the clipboard (handy!)

        Works for painsled (both iOS and desktop version) because they use
        the lapIdx column

        """

        if self.empty:
            return ([],[],[])

        df = self.df
        df['deltat'] = df['TimeStamp (sec)'].diff()

        workoutstateswork = [1, 4, 5, 8, 9, 6, 7]
        workoutstatesrest = [3]
        workoutstatetransition = [0, 2, 10, 11, 12, 13]

        intervalnrs = pd.unique(df[' lapIdx'])

        if debug:
            print('Interval numbers',intervalnrs)

        itime = []
        idist = []
        itype = []

        previousdist = 0.0
        # previoustime=0.0
        previoustime = df['TimeStamp (sec)'].min()

        try:
            test = df[' WorkoutState']
        except KeyError:
            df[' WorkoutState'] = 4

        for idx in intervalnrs:
            mask = df[' lapIdx'] == idx
            td = df[mask]

            # get stroke info
            mask = ~td[' WorkoutState'].isin(workoutstatesrest)
            tdwork = td[mask]
            mask = td[' WorkoutState'].isin(workoutstatesrest)
            tdrest = td[mask]

            try:
                # replaced ix with loc
                workoutstate = tdwork.loc[tdwork.index[-1],' WorkoutState']
            except IndexError:
                workoutstate = 4


            intervaldistance = tdwork['cum_dist'].dropna().max() - previousdist
            if isnan(intervaldistance) or isinf(intervaldistance):
                intervaldistance = 0

            if not isnan(td['cum_dist'].max()):
                previousdist = td['cum_dist'].max()


            intervalduration = tdwork['TimeStamp (sec)'].max() - previoustime
            # previoustime=tdrest[' ElapsedTime (sec)'].max()
            previoustime = td['TimeStamp (sec)'].max()

            intervalduration = nanstozero(intervalduration)
            restdistance = nanstozero(tdrest['cum_dist'].max())
            restdistance = restdistance - nanstozero(tdwork['cum_dist'].max())

            if restdistance < 0:
                restdistance = 0

            restduration = nanstozero(tdrest['TimeStamp (sec)'].max())
            restduration = restduration - \
                nanstozero(tdwork['TimeStamp (sec)'].max())
            if restduration < 0:
                restduration = 0


            #    if intervaldistance != 0:
            itime += [int(10 * intervalduration) / 10.,
                      int(10 * restduration) / 10.]
            idist += [int(intervaldistance),
                      int(restdistance)]
            itype += [workoutstate, 3]

        return itime, idist, itype

    def intervalstats_painsled(self, separator='|'):
        """ Used to create a nifty text summary, one row for each interval

        Also copies the string to the clipboard (handy!)

        Works for painsled (both iOS and desktop version) because they use
        the lapIdx column

        """

        if self.empty:
            return ""

        df = self.df
        df['deltat'] = df['TimeStamp (sec)'].diff()

        workoutstateswork = [1, 4, 5, 8, 9, 6, 7]
        workoutstatesrest = [3]
        workoutstatetransition = [0, 2, 10, 11, 12, 13]

        intervalnrs = pd.unique(df[' lapIdx'])

        stri = "Workout Details\n"
        stri += "#-{sep}SDist{sep}-Split-{sep}-SPace-{sep}-Pwr-{sep}SPM-{sep}AvgHR{sep}MaxHR{sep}DPS-\n".format(
            sep=separator
        )

        previousdist = 0.0
        # previoustime=0.0
        previoustime = df['TimeStamp (sec)'].min()

        try:
            test = df[' WorkoutState']
        except KeyError:
            return self.intervalstats()

        for index, idx in enumerate(intervalnrs):
            mask = df[' lapIdx'] == idx
            td = df[mask]

            # get stroke info
            mask = ~td[' WorkoutState'].isin(workoutstatesrest)
            tdwork = td[mask]
            mask = td[' WorkoutState'].isin(workoutstatesrest)
            tdrest = td[mask]

            try:
                workoutstate = tdwork.loc[tdwork.index[-1],' WorkoutState']
            except IndexError:
                workoutstate = 4


            #avghr = tdwork[' HRCur (bpm)'].mean()
            avghr = wavg(tdwork,' HRCur (bpm)','deltat')
            maxhr = tdwork[' HRCur (bpm)'].max()
            #avgspm = tdwork[' Cadence (stokes/min)'].mean()
            #avgpower = tdwork[' Power (watts)'].mean()
            avgspm = wavg(tdwork,' Cadence (stokes/min)','deltat')
            avgpower = wavg(tdwork,' Power (watts)','deltat')

            intervaldistance = tdwork['cum_dist'].max() - previousdist
            if isnan(intervaldistance) or isinf(intervaldistance):
                intervaldistance = 0

            if not isnan(td['cum_dist'].max()):
                previousdist = td['cum_dist'].max()

            intervalduration = tdwork['TimeStamp (sec)'].max() - previoustime
            # previoustime=tdrest[' ElapsedTime (sec)'].max()
            previoustime = td['TimeStamp (sec)'].max()

            if intervaldistance != 0:
                intervalpace = 500. * intervalduration / intervaldistance
            else:
                intervalpace = 0

            avgdps = intervaldistance / (intervalduration * avgspm / 60.)
            if isnan(avgdps) or isinf(avgdps):
                avgdps = 0
            if isnan(intervalpace) or isinf(intervalpace):
                intervalpace = 0
            if isnan(avgspm) or isinf(avgspm):
                avgspm = 0
            if isnan(avghr) or isinf(avghr):
                avghr = 0
            if isnan(maxhr) or isinf(maxhr):
                maxhr = 0

            if intervaldistance != 0:
                try:
                    stri += interval_string(intervalnrs[index],
                                            intervaldistance,
                                            intervalduration,
                                            intervalpace, avgspm,
                                            avghr, maxhr, avgdps, avgpower,
                                            separator=separator)
                except IndexError:
                    stri += interval_string(len(intervalnrs) + 1,
                                            intervaldistance,
                                            intervalduration,
                                            intervalpace, avgspm,
                                            avghr, maxhr, avgdps, avgpower,
                                            separator=separator)

        return stri

    def restoreintervaldata(self):

        try:
            self.df[' Horizontal (meters)'] = self.df['orig_dist']
            self.df['TimeStamp (sec)'] = self.df['orig_time']
            self.df[' ElapsedTime (sec)'] = self.df['orig_reltime']
            self.df[' LapIdx'] = self.df['orig_idx']
            self.df[' WorkoutState'] = self.df['orig_state']
        except KeyError:
            pass

    def updateintervaldata(self,
                           ivalues,
                           iunits,
                           itypes,
                           iresults=[],
                           debug=False,
                           ):
        """ Edits the intervaldata. For example a 2x2000m
        values=[2000,120,2000,120]
        units=['meters','seconds','meters','seconds']
        types=['work','rest','work','rest']
        """

        if self.empty:
            return None

        df = self.df
        try:
            origdist = df['orig_dist']
            df[' Horizontal (meters)'] = df['orig_dist']
            df['TimeStamp (sec)'] = df['orig_time']
            df[' ElapsedTime (sec)'] = df['orig_reltime']
            df[' LapIdx'] = df['orig_idx']
            df[' WorkoutState'] = df['orig_state']
        except KeyError:
            df['orig_dist'] = df[' Horizontal (meters)']
            df['orig_time'] = df['TimeStamp (sec)']
            df['orig_reltime'] = df[' ElapsedTime (sec)']
            df['orig_idx'] = df[' lapIdx']
            try:
                df['orig_state'] = df[' WorkoutState']
            except KeyError:
                df['orig_state'] = 1

        intervalnr = 0
        startmeters = 0
        # replaced ix with loc/iloc
        timezero = -df.loc[:, 'TimeStamp (sec)'].iloc[0] + \
            df.loc[:, ' ElapsedTime (sec)'].iloc[0]
        startseconds = 0

        endseconds = startseconds
        endmeters = startmeters

        if debug:
            print('Duration', df['TimeStamp (sec)'].max() - df['TimeStamp (sec)'].min())

        # erase existing lap data
        df[' lapIdx'] = 0
        df[' WorkoutState'] = 1
        df[' ElapsedTime (sec)'] = df['TimeStamp (sec)'] + timezero
        df[' Horizontal (meters)'] = df['cum_dist']

        for i in range(len(ivalues)):
            thevalue = ivalues[i]
            theunit = iunits[i]
            thetype = itypes[i]

            if debug:
                print(thevalue, theunit, thetype)

            if thetype == 'rest' and intervalnr != 0:
                intervalnr = intervalnr - 1

            workouttype = 1
            if theunit == 'meters' and thevalue > 0:
                workouttype = 5

                if thetype == 'rest':
                    workouttype = 3

                endmeters = startmeters + thevalue
                mask = (df['cum_dist'] > startmeters)
                df.loc[mask, ' lapIdx'] = intervalnr
                df.loc[mask, ' WorkoutState'] = workouttype
                df.loc[mask, ' ElapsedTime (sec)'] = df.loc[mask,
                                                            'TimeStamp (sec)'] - startseconds
                df.loc[mask, ' Horizontal (meters)'] = df.loc[mask,
                                                              'cum_dist'] - startmeters

                mask = (df['cum_dist'] <= endmeters)

                # correction for missing part of last stroke
                recordedmaxmeters = df.loc[mask, 'cum_dist'].max()
                deltadist = endmeters - recordedmaxmeters

                try:
                    res = iresults[i]
                    if not np.isnan(res):
                        mask2 = (df['cum_dist'] == recordedmaxmeters)
                        if res == 0:
                            raise IndexError
                        deltatime = res - df.loc[mask2, ' ElapsedTime (sec)']
                        mask2 = (df['cum_dist'] == recordedmaxmeters)
                        df.loc[mask2, ' ElapsedTime (sec)'] = res
                        df.loc[mask2, 'TimeStamp (sec)'] += deltatime
                        df.loc[mask2, ' Horizontal (meters)'] += deltadist
                except IndexError:
                    if deltadist > 25:
                        deltadist = 0
                    mask2 = (df['cum_dist'] == recordedmaxmeters)
                    try:
                        paceend = df.loc[mask2,
                                         ' Stroke500mPace (sec/500m)'
                        ].values[0]
                        veloend = 500. / paceend
                        deltatime = deltadist / veloend
                    except IndexError:
                        deltatime = 0

                    df.loc[mask2, ' ElapsedTime (sec)'] += deltatime
                    df.loc[mask2, 'TimeStamp (sec)'] += deltatime
                    df.loc[mask2, ' Horizontal (meters)'] += deltadist
                    df.loc[mask2, 'cum_dist'] += deltadist

                # + deltatime?
                endseconds = df.loc[mask, 'TimeStamp (sec)'].max()

            if theunit == 'seconds' and thevalue > 0:
                workouttype = 4

                if thetype == 'rest':
                    workouttype = 3

                endseconds = startseconds + thevalue

                mask = (df['TimeStamp (sec)'] > startseconds)
                if startseconds == 0:
                    mask = (df['TimeStamp (sec)'] >= startseconds)
                df.loc[mask, ' lapIdx'] = intervalnr
                df.loc[mask, ' WorkoutState'] = workouttype
                df.loc[mask, ' ElapsedTime (sec)'] = df.loc[mask,
                                                            'TimeStamp (sec)'] - startseconds
                df.loc[mask, ' Horizontal (meters)'] = df.loc[mask,
                                                              'cum_dist'] - startmeters

                mask = (df['TimeStamp (sec)'] <= endseconds)

                # correction for missing part of last stroke
                recordedmaxtime = df.loc[mask, 'TimeStamp (sec)'].max()
                deltatime = endseconds - recordedmaxtime
                try:
                    res = iresults[i]
                    if not np.isnan(res):
                        mask2 = (df['TimeStamp (sec)'] == recordedmaxtime)
                        deltadist = res - df.loc[mask2, ' Horizontal (meters)']
                        if res == 0:
                            raise IndexError
                        mask2 = (df['TimeStamp (sec)'] == recordedmaxtime)
                        df.loc[mask2, ' ElapsedTime (sec)'] += deltatime
                        df.loc[mask2, ' Horizontal (meters)'] = res
                        df.loc[mask2, 'TimeStamp (sec)'] += deltatime
                        df.loc[mask2, 'cum_dist'] += deltadist
                except IndexError:
                    if deltatime > 6 and thetype != 'rest':
                        deltatime = 0
                    mask2 = (df['TimeStamp (sec)'] == recordedmaxtime)
                    paceend = df.loc[mask2,
                                     ' Stroke500mPace (sec/500m)'].values[0]
                    veloend = 500. / paceend
                    deltadist = veloend * deltatime
                    if deltatime > 5 and thetype == 'rest':
                        deltadist = 0

                    df.loc[mask2, ' ElapsedTime (sec)'] += deltatime
                    df.loc[mask2, ' Horizontal (meters)'] += deltadist
                    df.loc[mask2, 'cum_dist'] += deltadist
                    df.loc[mask2, 'TimeStamp (sec)'] += deltatime

                mask = (df['TimeStamp (sec)'] <= endseconds)

                endmeters = df.loc[mask, 'cum_dist'].max()  # + deltadist?

            intervalnr += 1

            startseconds = endseconds
            startmeters = endmeters

            if debug:
                print(intervalnr, startseconds, startmeters)

        self.df = df

    def updateinterval_metric(self, metricname, value, debug=False,
                              mode='split',unit='seconds',
                              smoothwindow = 60.,
                              activewindow = []):
        if self.empty:
            return None

        df = self.df

        if activewindow == []:
            activewindow = [0,self.duration]

        try:
            origdist = df['orig_dist']
            df[' Horizontal (meters)'] = df['orig_dist']
            df['TimeStamp (sec)'] = df['orig_time']
            df[' ElapsedTime (sec)'] = df['orig_reltime']
            df[' LapIdx'] = df['orig_idx']
            df[' WorkoutState'] = df['orig_state']
        except KeyError:
            df['orig_dist'] = df[' Horizontal (meters)']
            df['orig_time'] = df['TimeStamp (sec)']
            df['orig_reltime'] = df[' ElapsedTime (sec)']
            df['orig_idx'] = df[' lapIdx']
            try:
                df['orig_state'] = df[' WorkoutState']
            except KeyError:
                df['orig_state'] = 1

        # replaced ix with loc/iloc
        timezero = -df.loc[:, 'TimeStamp (sec)'].iloc[0] + \
            df.loc[:, ' ElapsedTime (sec)'].iloc[0]

        # erase existing lap data
        df[' lapIdx'] = 0
        df[' WorkoutState'] = 5
        df[' ElapsedTime (sec)'] = df['TimeStamp (sec)'] + timezero
        df[' Horizontal (meters)'] = df['cum_dist']


        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df[' AverageBoatSpeed (m/s)'] = df[' AverageBoatSpeed (m/s)'].replace(np.nan,0)

        df = df.fillna(method='bfill',axis=0)
        self.df = df

        values = self.get_smoothed(metricname,smoothwindow)

        if mode == 'larger':
            largerthantype = 5
            smallerthantype = 3
        else:
            largerthantype = 3
            smallerthantype = 5


        mask = (values > value)
        df.loc[mask, ' WorkoutState'] = largerthantype
        mask = (values <= value)
        df.loc[mask, ' WorkoutState'] = smallerthantype

        # do rest for begin and end
        mask = (df[' ElapsedTime (sec)'] < activewindow[0] )
        df.loc[mask, ' WorkoutState'] = 3
        mask =  (df[' ElapsedTime (sec)'] > activewindow[1])
        df.loc[mask, ' WorkoutState'] = 3

        steps = df[' WorkoutState'].diff()

        indices = df.index[steps!=0].tolist()
        if debug:
            print('indices ',indices)
            print('----------------------')

        intervalnr = 0

        for i in range(len(indices[1:])):
            if debug:
                print(indices[i+1]-indices[i])
            # replacing ix with loc/iloc
            df.loc[:,' lapIdx'].iloc[indices[i]:indices[i+1]] = intervalnr
            intervalnr += 1

        # replacing ix with loc/iloc
        df.loc[:,' lapIdx'].iloc[indices[-1]:] = intervalnr
        df['values'] = (1+df[' lapIdx'])*10 + df[' WorkoutState']

        valuecounts = Counter(df['values'])
        if debug:
            print(valuecounts)
            print('----------------------------')


        f = df['TimeStamp (sec)'].diff().mean()

        tenstrokes = int(25/f)
        if debug:
            print('Ten Strokes = ',tenstrokes,' data points')

        for key, value in valuecounts.items():
            if value < tenstrokes:
                if debug:
                    print(key,value)
                mask = df['values'] == key
                df.loc[mask,' WorkoutState'] = np.nan

        df = df.fillna(method='ffill',axis=0)
        self.df = df

        df[' lapIdx'] = 0

        steps = df[' WorkoutState'].diff()

        indices = df.index[steps!=0].tolist()


        if debug:
            print('indices ',indices)
            print('----------------------')

        intervalnr = 0


        if unit == 'meters':
            elapsemetric = 'cum_dist'
        else:
            elapsemetric = 'TimeStamp (sec)'


        previouselapsed = df.loc[indices[0],elapsemetric] # replaced ix with loc

        units = []
        typ = []
        vals = []

        for i in indices[1:]:
            try:
                startindex = df.index[i-1]
            except KeyError:
                startindex = 0


            if debug:
                print(df.loc[startindex,'cum_dist']) # replaced ix with loc

            startelapsed = df.loc[startindex,elapsemetric] # replaced ix with loc

            units.append(unit)
            vals.append(startelapsed-previouselapsed)

            if df.loc[startindex,' WorkoutState'] == 3: # replaced ix with loc
                tt = 'rest'
            else:
                tt = 'work'

            if mode == 'split':
                tt = 'work'

            typ.append(tt)

            if debug:
                print(startindex,startelapsed-previouselapsed,unit,tt)

            previouselapsed = startelapsed

        # final part
        startindex = df.index[-1]
        startelapsed = df.loc[startindex,elapsemetric] # replaced ix with loc

        if debug:
            print(df.loc[startindex,'cum_dist']) # replaced ix with loc

        units.append(unit)
        vals.append(startelapsed-previouselapsed)

        if df.loc[startindex,' WorkoutState'] == 3: # replaced ix with loc
            tt = 'rest'
        else:
            tt = 'work'

        if mode == 'split':
            tt = 'work'

        typ.append(tt)

        if debug:
            print(startindex,startelapsed-previouselapsed,unit,tt)

        if debug:
            print('--------------------------------')

        self.df = df
        self.updateintervaldata(vals,units,typ,debug=debug)


        if debug:
            print(vals)
            print(units)
            print(typ)

        if mode == 'split':
            self.df[' WorkoutState'] = 5



    def updateinterval_string(self, s, debug=False):
        res = trainingparser.parse(s)
        res = trainingparser.cleanzeros(res)
        if res:
            values = trainingparser.getlist(res)
            units = trainingparser.getlist(res, sel='unit')
            typ = trainingparser.getlist(res, sel='type')

            self.updateintervaldata(values, units, typ, debug=debug)

    def add_bearing(self, window_size=20):
        """ Adds bearing. Only works if long and lat values are known

        """
        nr_of_rows = self.df.shape[0]
        df = self.df

        bearing = np.zeros(nr_of_rows)

        # replacing ix with loc below
        for i in range(len(df.index)-1):
            index = df.index[i]
            try:
                long1 = df.loc[index, ' longitude']
                lat1 = df.loc[index, ' latitude']
                long2 = df.loc[:, ' longitude'].iloc[i+1]
                lat2 = df.loc[:, ' latitude'].iloc[i+1]
            except KeyError:
                long1 = 0
                lat1 = 0
                long2 = 0
                lat2 = 0
            res = geo_distance(lat1, long1, lat2, long2)
            bearing[i] = res[1]

        bearing2 = ewmovingaverage(bearing, window_size)

        df['bearing'] = 0
        df['bearing'] = bearing2

        self.df = df

        return 1

    def add_stream(self, vstream, units='m'):
        # foot/second
        if (units == 'f'):
            vstream = 0.3048 * vstream

        # knots
        if (units == 'k'):
            vstream = vstream / 1.994

        # pace difference (approximate)
        if (units == 'p'):
            vstream = vstream * 8 / 500.

        df = self.df

        df['vstream'] = vstream

        self.df = df

    def add_wind(self, vwind, winddirection, units='m'):

        # beaufort
        if (units == 'b'):
            vwind = 0.837 * vwind**(3. / 2.)
        # knots
        if (units == 'k'):
            vwind = vwind * 1.994

        # km/h
        if (units == 'kmh'):
            vwind = vwind / 3.6

        # mph
        if (units == 'mph'):
            vwind = 0.44704 * vwind

        df = self.df

        df['vwind'] = vwind
        df['winddirection'] = winddirection

        self.df = df

    def update_stream(self, stream1, stream2, dist1, dist2, units='m'):
        try:
            vs = self.df.loc[:, 'vstream'] # replaced ix with loc
        except KeyError:
            self.add_stream(0)

        df = self.df

        # foot/second
        if (units == 'f'):
            stream1 = 0.3048 * stream1
            stream2 = 0.3048 * stream2

        # knots
        if (units == 'k'):
            stream1 = stream1 / 1.994
            stream2 = stream2 / 1.994

        # pace difference (approximate)
        if (units == 'p'):
            stream1 = stream1 * 8 / 500.
            stream2 = stream2 * 8 / 500.

        aantal = len(df)

        # replaced ix with loc below
        for i in df.index:
            if (df.loc[i, 'cum_dist'] > dist1 and df.loc[i, 'cum_dist'] < dist2):
                # doe iets
                x = df.loc[i, 'cum_dist']
                r = (x - dist1) / (dist2 - dist1)
                stream = stream1 + (stream2 - stream1) * r
                try:
                    df.loc[i, 'vstream'] = stream
                except:
                    pass

        self.df = df

    def get_smoothed(self, metricname, windowseconds):
        if metricname == ' Stroke500mPace (sec/500m)':
            pace = self.df[' Stroke500mPace (sec/500m)'].values
            thevalues = 500. / pace
        else:
            thevalues = self.df[metricname].values

        f = self.df['TimeStamp (sec)'].diff().mean()
        if f!= 0 and not np.isnan(f):
            windowsize = 2 * (int(windowseconds/(f))) + 1
        else:
            windowsize = 1

        if windowsize > 3 and windowsize < len(thevalues):
            newvalues = savgol_filter(thevalues, windowsize, 3)
        else:
            newvalues = thevalues

        newvalues = pd.Series(newvalues)
        newvalues = newvalues.replace([-np.inf,np.inf],np.nan)
        newvalues = newvalues.fillna(method='ffill')

        if metricname == ' Stroke500mPace (sec/500m)':
            newvalues = 500./newvalues

        return newvalues

    def update_wind(self, vwind1, vwind2, winddirection1,
                    winddirection2, dist1, dist2, units='m'):

        try:
            vw = self.df.loc[:, 'vwind'] # replaced ix with loc
        except KeyError:
            self.add_wind(0, 0)

        df = self.df

        # beaufort
        if (units == 'b'):
            vwind1 = 0.837 * vwind1**(3. / 2.)
            vwind2 = 0.837 * vwind2**(3. / 2.)

        # knots
        if (units == 'k'):
            vwind1 = vwind1 / 1.994
            vwind2 = vwind2 / 1.994

        # km/h
        if (units == 'kmh'):
            vwind1 = vwind1 / 3.6
            vwind2 = vwind2 / 3.6

        # mph
        if (units == 'mph'):
            vwind1 = 0.44704 * vwind1
            vwind2 = 0.44704 * vwind2

        aantal = len(df)

        # replaced ix with loc below
        for i in df.index:
            if (df.loc[i, 'cum_dist'] > dist1 and df.loc[i, 'cum_dist'] < dist2):
                # doe iets
                x = df.loc[i, 'cum_dist']
                r = (x - dist1) / (dist2 - dist1)
                try:
                    vwind = vwind1 + (vwind2 - vwind1) * r
                    df.loc[i, 'vwind'] = vwind
                except:
                    pass
                try:
                    dirwind = winddirection1 + \
                        (winddirection2 - winddirection1) * r
                    df.loc[i, 'winddirection'] = dirwind
                except:
                    pass

        self.df = df

    def otw_setpower(self, skiprows=1, rg=getrigging(), mc=70.0,
                     powermeasured=False,
                     secret=None,progressurl=None,
                     usetable=False,storetable=None,
                     silent=False):
        """ Adds power from rowing physics calculations to OTW result

        For now, works only in singles

        """

        if self.empty:
            return None

        nr_of_rows = self.number_of_rows
        rows_mod = skiprows + 1
        df = self.df
        df['nowindpace'] = 300
        df['equivergpower'] = 0
        df['power (model)'] = 0
        df['averageforce (model)'] = 0
        df['drivelength (model)'] = 0

        # creating a rower and rigging for now
        # in future this must come from rowingdata.rower and rowingdata.rigging
        r = self.rwr.rc
        r.mc = mc

        # modify pace/spm/wind with rolling averages
        try:
            ps = df[' Stroke500mPace (sec/500m)'].rolling(skiprows+1).mean()
            spms = df[' Cadence (stokes/min)'].rolling(skiprows+1).mean()
        except AttributeError:
            ps = df[' Stroke500mPace (sec/500m)']
            spms = df[' Cadence (stokes/min)']

        if storetable is not None:
            try:
                if storetable[-3:] != 'npz':
                    filename = storetable+'.npz'
                else:
                    filename = storetable

                loaded = np.load(filename)
                T = loaded['T']
                S = loaded['S']
                try:
                    C = loaded['C']
                except KeyError:
                    C = np.zeros((Nspm,Nvw,Nvb))

                loaded.close()
            except IOError:
                T = np.zeros((Nspm,Nvw,Nvb))
                S = np.zeros((Nspm,Nvw,Nvb))
                C = np.zeros((Nspm,Nvw,Nvb))
        else:
            T = np.zeros((Nspm,Nvw,Nvb))
            S = np.zeros((Nspm,Nvw,Nvb))
            C = np.zeros((Nspm,Nvw,Nvb))

        # this is slow ... need alternative (read from table)

        # this is slow ... need alternative (read from table)
        counterrange = int(nr_of_rows/100.)
        counter = 0

        iterator = list(range(nr_of_rows))
        if not silent:
            iterator = tqdm(iterator)

        for i in iterator:
            counter += 1
            if counter>counterrange:
                counter = 0
                progress = int(100.*i/float(nr_of_rows))
                if secret and progressurl:
                    status_code = post_progress(secret,progressurl,progress)

            p = ps.iloc[i] # replaced ix with iloc
            spm = spms.iloc[i] # replaced ix with iloc
            r.tempo = spm
            try:
                drivetime = 60. * 1000. / float(spm)  # in milliseconds
            except ZeroDivisionError:
                drivetime = 4000.
            if (p != 0) & (spm != 0) & (p < 210):
                velo = 500. / p
                try:
                    # replaced ix with loc/iloc below
                    vwind = df.loc[:, 'vwind'].iloc[i]
                    winddirection = df.loc[:, 'winddirection'].iloc[i]
                    bearing = df.loc[:, 'bearing'].iloc[i]
                except KeyError:
                    vwind = 0.0
                    winddirection = 0.0
                    bearing = 0.0
                try:
                    # replaced ix with loc/iloc
                    vstream = df.loc[:, 'vstream'].iloc[i]
                except KeyError:
                    vstream = 0

                if (i % rows_mod == 0):
                    tw = tailwind(bearing, vwind,
                                  winddirection, vstream=0)
                    velowater = velo - vstream

                    u,v,w = getaddress(spm, tw, velowater)
                    if usetable:
                        pwr = T[u,v,w]
                        nowindpace = S[u,v,w]
                    else:
                        pwr = -1

                    if pwr > 0:
                        res = [pwr,np.nan,np.nan,nowindpace,np.nan]
                    else:
                        try:
                            res = phys_getpower(velo, r, rg,
                                                bearing, vwind,
                                                winddirection,
                                                vstream)
                            if usetable:
                                T[u,v,w] = res[0]
                                S[u,v,w] = res[3]
                                C[u,v,w] += 1
                            if storetable is not None:
                                if not usetable:
                                    count = float(C[u,v,w])
                                    T[u,v,w] = (count*T[u,v,w]+res[0])/(count+1.0)
                                    S[u,v,w] = (count*S[u,v,w]+res[3])/(count+1.0)
                                    C[u,v,w] += 1.

                        except:
                            res = [np.nan, np.nan, np.nan, np.nan, np.nan]
                else:
                    res = [np.nan, np.nan, np.nan, np.nan, np.nan]
                if not np.isnan(res[0]) and res[0] < 800:
                    df.loc[:, 'power (model)'].iloc[i] = res[0] # ix -> loc/iloc
                else:
                    df.loc[:, 'power (model)'].iloc[i] = np.nan # ix -> loc/iloc
                # replacing ix with loc/iloc below
                df.loc[:, 'averageforce (model)'].iloc[i] = res[2] / lbstoN
                df.loc[:, ' DriveTime (ms)'].iloc[i] = res[1] * drivetime
                df.loc[:, ' StrokeRecoveryTime (ms)'].iloc[i] = (1 - res[1]) * drivetime
                df.loc[:, 'drivelength (model)'].iloc[i] = r.strokelength
                df.loc[:, 'nowindpace'].iloc[i] = res[3]
                df.loc[:, 'equivergpower'].iloc[i] = res[4]

                if (res[4] > res[0]) and not silent:
                    print(("Power ", res[0]))
                    print(("Equiv erg Power ", res[4]))
                    print(("Boat speed (m/s) ", velo))
                    print(("Stroke rate ", r.tempo))
                    print(("ratio ", res[1]))
                # update_progress(i,nr_of_rows)

            else:
                velo = 0.0

        if storetable is not None:
            np.savez_compressed(storetable,T=T,S=S,C=C)

        self.df = df.interpolate()
        if not powermeasured:
            self.df[' Power (watts)'] = self.df['power (model)']
            self.df[' AverageDriveForce (lbs)'] = self.df['averageforce (model)']
            self.df[' DriveLength (meters)'] = self.df['drivelength (model)']

    def otw_setpower_silent(self, skiprows=1, rg=getrigging(), mc=70.0,
                            powermeasured=False,
                            secret=None,progressurl=None,
                            usetable=False,storetable=None):

        """ Adds power from rowing physics calculations to OTW result

        For now, works only in singles

        """
        if self.empty:
            return None

        if (weknowphysics != 1):
            return None

        nr_of_rows = self.number_of_rows
        rows_mod = skiprows + 1
        df = self.df
        df['nowindpace'] = 300
        df['equivergpower'] = 0
        df['power (model)'] = 0
        df['averageforce (model)'] = 0
        df['drivelength (model)'] = 0

        # creating a rower and rigging for now
        # in future this must come from rowingdata.rower and rowingdata.rigging
        r = self.rwr.rc
        r.mc = mc


        # modify pace/spm/wind with rolling averages
        try:
            ps = df[' Stroke500mPace (sec/500m)'].rolling(skiprows+1).mean()
            spms = df[' Cadence (stokes/min)'].rolling(skiprows+1).mean()
        except AttributeError:
            ps = df[' Stroke500mPace (sec/500m)']
            spms = df[' Cadence (stokes/min)']

        if storetable is not None:
            try:
                if storetable[-3:] != 'npz':
                    filename = storetable+'.npz'
                else:
                    filename = storetable

                loaded = np.load(filename)

                print('loaded %s' % filename)

                T = loaded['T']
                S = loaded['S']

                try:
                    C = loaded['C']
                except KeyError:
                    C = np.zeros((Nspm,Nvw,Nvb))

                print('Non zero %d ' % np.count_nonzero(T))

                loaded.close()
            except IOError:
                T = np.zeros((Nspm,Nvw,Nvb))
                S = np.zeros((Nspm,Nvw,Nvb))
                C = np.zeros((Nspm,Nvw,Nvb))
            else:
                T = np.zeros((Nspm,Nvw,Nvb))
                S = np.zeros((Nspm,Nvw,Nvb))
                C = np.zeros((Nspm,Nvw,Nvb))

        # this is slow ... need alternative (read from table)
        counterrange = int(nr_of_rows/100.)
        counter = 0
        counter2 = 0
        for i in range(nr_of_rows):
            counter += 1
            if counter>counterrange:
                counter = 0
                progress = int(100.*i/float(nr_of_rows))
                if secret and progressurl:
                    status_code = post_progress(secret,progressurl,progress)

            p = ps.iloc[i]
            spm = spms.iloc[i]
            r.tempo = spm

            try:
                drivetime = 60. * 1000. / float(spm)  # in milliseconds
            except ZeroDivisionError:
                drivetime = 4000.
            if (p != 0) & (spm != 0) & (p < 210):
                velo = 500. / p
                try:
                    vwind = df.loc[:, 'vwind'].iloc[i] # ix -> loc/iloc
                    winddirection = df.loc[:, 'winddirection'].iloc[i] # ix -> loc/ilic
                    bearing = df.loc[:, 'bearing'].iloc[i] # ix -> loc/ilic
                except KeyError:
                    vwind = 0.0
                    winddirection = 0.0
                    bearing = 0.0
                try:
                    vstream = df.loc[:, 'vstream'].iloc[i] # ix -> loc/ilic
                except KeyError:
                    vstream = 0

                if (i % rows_mod == 0):

                    print('Task %s: working on row %s of %s ' % (counter2,i,nr_of_rows))

                    tw = tailwind(bearing, vwind,
                                  winddirection, vstream=0)
                    velowater = velo - vstream

                    u,v,w = getaddress(spm, tw, velowater)

                    if usetable:
                        pwr = T[u,v,w]
                        nowindpace = S[u,v,w]
                    else:
                        pwr = -1

                    if pwr > 0:
                        res = [pwr,np.nan,np.nan,nowindpace,np.nan]
                    else:
                        counter2 += 1
                        try:
                            res = phys_getpower(velo, r, rg,
                                                bearing, vwind,
                                                winddirection,
                                                vstream)
                            if usetable:
                                T[u,v,w] = res[0]
                                S[u,v,w] = res[3]
                                C[u,v,w] += 1
                            if storetable is not None:
                                if not usetable:
                                    count = float(C[u,v,w])
                                    T[u,v,w] = (count*T[u,v,w]+res[0])/count
                                    S[u,v,w] = (count*S[u,v,w]+res[0])/count
                                    C[u,v,w] += 1.
                        except:
                            res = [np.nan, np.nan, np.nan, np.nan, np.nan]
                else:
                    res = [np.nan, np.nan, np.nan, np.nan, np.nan]
                if not np.isnan(res[0]) and res[0] < 800:
                    df.loc[:, 'power (model)'].iloc[i] = res[0] # ix -> loc/ilic
                else:
                    df.loc[:, 'power (model)'].iloc[i] = np.nan # ix -> loc/ilic

                # replacing ix with loc/iloc below
                df.loc[:, 'averageforce (model)'].iloc[i] = res[2] / lbstoN
                df.loc[:, ' DriveTime (ms)'].iloc[i] = res[1] * drivetime
                df.loc[:, ' StrokeRecoveryTime (ms)'].iloc[i] = (1 - res[1]) * drivetime
                df.loc[:, 'drivelength (model)'].iloc[i] = r.strokelength
                df.loc[:, 'nowindpace'].iloc[i] = res[3]
                df.loc[:, 'equivergpower'].iloc[i] = res[4]
                # update_progress(i,nr_of_rows)

            else:
                velo = 0.0

        if storetable is not None:
            np.savez_compressed(storetable,T=T,S=S,C=C)

        self.df = df.interpolate()
        if not powermeasured:
            self.df[' Power (watts)'] = self.df['power (model)']
            self.df[' AverageDriveForce (lbs)'] = self.df['averageforce (model)']
            self.df[' DriveLength (meters)'] = self.df['drivelength (model)']

        return 1

    def otw_setpower_verbose(self, skiprows=0, rg=getrigging(), mc=70.0,
                             powermeasured=False):
        """ Adds power from rowing physics calculations to OTW result

        For now, works only in singles

        """
        if self.empty:
            return None

        print("EXPERIMENTAL")

        nr_of_rows = self.number_of_rows
        rows_mod = skiprows + 1
        df = self.df
        df['nowindpace'] = 300
        df['equivergpower'] = 0
        df['power (model)'] = 0
        df['averageforce (model)'] = 0
        df['drivelength (model)'] = 0

        # creating a rower and rigging for now
        # in future this must come from rowingdata.rower and rowingdata.rigging
        r = self.rwr.rc
        r.mc = mc

        # this is slow ... need alternative (read from table)
        for i in range(nr_of_rows):
            p = df.loc[:, ' Stroke500mPace (sec/500m)'].iloc[i] # ix -> loc/iloc
            spm = df.loc[:, ' Cadence (stokes/min)'].iloc[i] # ix -> loc/iloc
            r.tempo = spm
            try:
                drivetime = 60. * 1000. / float(spm)  # in milliseconds
            except ZeroDivisionError:
                drivetime = 4000.
            if (p != 0) & (spm != 0) & (p < 210):
                velo = 500. / p
                try:
                    vwind = df.loc[:, 'vwind'].iloc[i] # ix -> loc/iloc
                    winddirection = df.loc[:, 'winddirection'].iloc[i] # ix -> loc/iloc
                    bearing = df.loc[:, 'bearing'].iloc[i] # ix -> loc/iloc
                except KeyError:
                    vwind = 0.0
                    winddirection = 0.0
                    bearing = 0.0
                try:
                    vstream = df.loc[:, 'vstream'].iloc[i] # ix -> loc/iloc
                except KeyError:
                    vstream = 0

                if (i % rows_mod == 0):
                    try:
                        res = phys_getpower(velo, r, rg, bearing, vwind, winddirection,
                                            vstream)
                        print((i, r.tempo, p, res[0], res[3], res[4]))
                    except KeyError:
                        res = [np.nan, np.nan, np.nan, np.nan, np.nan]
                else:
                    res = [np.nan, np.nan, np.nan, np.nan, np.nan]
                # replacing ix with loc/iloc
                df.loc[:, 'power (model)'].iloc[i] = res[0]
                df.loc[:, 'averageforce (model)'].iloc[i] = res[2] / lbstoN
                df.loc[:, ' DriveTime (ms)'] = res[1].iloc[i] * drivetime
                df.loc[:, ' StrokeRecoveryTime (ms)'].iloc[i] = (1 - res[1]) * drivetime
                df.loc[:, 'drivelength (model)'].iloc[i] = r.strokelength
                df.loc[:, 'nowindpace'].iloc[i] = res[3]
                df.loc[:, 'equivergpower'].iloc[i] = res[4]
                # update_progress(i,nr_of_rows)
            else:
                velo = 0.0

        self.df = df.interpolate()
        if not powermeasured:
            self.df[' Power (watts)'] = self.df['power (model)']
            self.df[' AverageDriveForce (lbs)'] = self.df['averageforce (model)']
            self.df[' DriveLength (meters)'] = self.df['drivelength (model)']

    def otw_testphysics(self, rg=getrigging(), mc=70.0, p=120., spm=30.):
        """ Check if erg pace is in right order

        For now, works only in singles

        """
        if self.empty:
            return None

        print("EXPERIMENTAL")

        # creating a rower and rigging for now
        # in future this must come from rowingdata.rower and rowingdata.rigging
        r = self.rwr.rc
        r.mc = mc
        r.tempo = spm
        drivetime = 60. * 1000. / float(spm)  # in milliseconds
        if (p != 0) & (spm != 0) & (p < 210):
            velo = 500. / p
            vwind = 0.0
            winddirection = 0.0
            bearing = 0.0
            vstream = 0.0
            res = phys_getpower(velo, r, rg, bearing, vwind, winddirection,
                                vstream)

            print(('Pace ', p))
            print(('Power (watts)', res[0]))
            print(('Average Drive Force (N)', res[2]))
            print((' DriveTime (ms)', res[1] * drivetime))
            print((' StrokeRecoveryTime (ms)', (1 - res[1]) * drivetime))
            print((' DriveLength (meters)', r.strokelength))
            print(('nowindpace', res[3]))
            print(('equivergpower', res[4]))

        else:
            velo = 0.0

    def summary(self, separator='|'):
        """ Creates a nifty text string that contains the key data for the row
        and copies it to the clipboard

        """
        if self.empty:
            return ""

        df = self.df
        df['deltat'] = df['TimeStamp (sec)'].diff()

        # total dist, total time, avg pace, avg hr, max hr, avg dps

        times, distances, types = self.intervalstats_values()

        times = np.array(times)
        distance = np.array(distances)
        types = np.array(types)

        totaldist = np.array(distances).sum()
        totaltime = np.array(times).sum()

        avgpace = 500 * totaltime / totaldist
        #avghr = df[' HRCur (bpm)'].mean()
        maxhr = df[' HRCur (bpm)'].max()
        #avgspm = df[' Cadence (stokes/min)'].mean()
        #avgpower = df[' Power (watts)'].mean()

        avghr = wavg(df,' HRCur (bpm)','deltat')

        avgspm = wavg(df,' Cadence (stokes/min)','deltat')
        avgpower = wavg(df,' Power (watts)','deltat')
        avgdps = 0
        if totaltime * avgspm > 0:
            avgdps = totaldist / (totaltime * avgspm / 60.)

        stri = summarystring(totaldist, totaltime, avgpace, avgspm,
                             avghr, maxhr, avgdps, avgpower,
                             readFile=self.readfilename,
                             separator=separator)

        try:
            test = df[' WorkoutState']
        except KeyError:
            return stri

        workoutstateswork = [1, 4, 5, 8, 9, 6, 7]
        workoutstatesrest = [3]
        workoutstatetransition = [0, 2, 10, 11, 12, 13]

        intervalnrs = pd.unique(df[' lapIdx'])

        previousdist = 0.0
        # previoustime=0.0
        previoustime = df['TimeStamp (sec)'].min()

        workttot = 0.0
        workdtot = 0.0

        workspmavg = 0
        workhravg = 0
        workdpsavg = 0
        workhrmax = 0
        workpoweravg = 0

        restttot = 0.0
        restdtot = 0.0

        restspmavg = 0
        resthravg = 0
        restdpsavg = 0
        restpoweravg = 0
        resthrmax = 0

        for idx in intervalnrs:
            td = df[df[' lapIdx'] == idx]

            # get stroke info
            tdwork = td[~td[' WorkoutState'].isin(workoutstatesrest)]
            tdrest = td[td[' WorkoutState'].isin(workoutstatesrest)]

            avghr = nanstozero(wavg(tdwork,' HRCur (bpm)','deltat'))
            maxhr = nanstozero(tdwork[' HRCur (bpm)'].max())
            avgspm = nanstozero(wavg(tdwork,' Cadence (stokes/min)','deltat'))
            avgpower = nanstozero(wavg(tdwork,' Power (watts)','deltat'))

            avghrrest = nanstozero(wavg(tdrest,' HRCur (bpm)','deltat'))
            maxhrrest = nanstozero(tdrest[' HRCur (bpm)'].max())
            avgspmrest = nanstozero(wavg(tdrest,' Cadence (stokes/min)','deltat'))
            avgrestpower = nanstozero(wavg(tdrest,' Power (watts)','deltat'))

            intervaldistance = tdwork['cum_dist'].max() - previousdist
            if isnan(intervaldistance) or isinf(intervaldistance):
                intervaldistance = 0

            intervalduration = nanstozero(
                tdwork['TimeStamp (sec)'].max() - previoustime)

            previoustime = td['TimeStamp (sec)'].max()

            restdistance = tdrest['cum_dist'].max() - tdwork['cum_dist'].max()
            if np.isnan(tdwork['cum_dist'].max()):
                restdistance = tdrest['cum_dist'].max() - previousdist

            restdistance = nanstozero(restdistance)
            previousdist = td['cum_dist'].max()

            restduration = nanstozero(tdrest[' ElapsedTime (sec)'].max()-tdwork['TimeStamp (sec)'].max())
            if restduration<=0:
                restduration = nanstozero(tdrest[' ElapsedTime (sec)'].max())

            if intervaldistance != 0:
                intervalpace = 500. * intervalduration / intervaldistance
            else:
                intervalpace = 0

            if restdistance > 0:
                restpace = 500. * restduration / restdistance
            else:
                restpace = 0

            if (intervalduration * avgspm > 0):
                avgdps = intervaldistance / (intervalduration * avgspm / 60.)
            else:
                avgdps = 0

            if (restduration * avgspmrest > 0):
                restdpsavg = restdistance / (restduration * avgspmrest / 60.)
            else:
                restdpsavg = 0
            if isnan(avgdps) or isinf(avgdps):
                avgdps = 0

            if isnan(restdpsavg) or isinf(restdpsavg):
                restdpsavg = 0

            workspmavg = workspmavg * workttot + intervalduration * avgspm
            workhravg = workhravg * workttot + intervalduration * avghr
            workdpsavg = workdpsavg * workttot + intervalduration * avgdps
            workpoweravg = workpoweravg * workttot + intervalduration * avgpower
            if workttot + intervalduration > 0:
                workspmavg = workspmavg / (workttot + intervalduration)
                workhravg = workhravg / (workttot + intervalduration)
                workdpsavg = workdpsavg / (workttot + intervalduration)
                workpoweravg = workpoweravg / (workttot + intervalduration)

            workhrmax = max(workhrmax, maxhr)

            restspmavg = restspmavg * restttot + restduration * avgspmrest
            resthravg = resthravg * restttot + restduration * avghrrest
            restdpsavg = restdpsavg * restttot + restduration * restdpsavg
            restpoweravg = restpoweravg * restttot + restduration * avgrestpower

            if restttot + restduration > 0:
                restspmavg = restspmavg / (restttot + restduration)
                resthravg = resthravg / (restttot + restduration)
                restdpsavg = restdpsavg / (restttot + restduration)
                restpoweravg = restpoweravg / (restttot + restduration)

            resthrmax = max(resthrmax, maxhr)

            workttot += intervalduration
            workdtot += intervaldistance

            restttot += restduration
            restdtot += restdistance

        if restdtot != 0:
            avgrestpace = 500. * restttot / restdtot
        else:
            avgrestpace = 0

        if workdtot != 0:
            avgworkpace = 500. * workttot / workdtot
        else:
            avgworkpace = 1000.

        stri += workstring(workdtot, workttot, avgworkpace, workspmavg,
                           workhravg, workhrmax, workdpsavg, workpoweravg,
                           separator=separator,
                           symbol='W')


        stri += workstring(restdtot, restttot, avgrestpace, restspmavg,
                           resthravg, resthrmax, restdpsavg, restpoweravg,
                           separator=separator,
                           symbol='R')

        return stri

    def allstats(self, separator='|'):
        """ Creates a nice text summary, both overall summary and a one line
        per interval summary

        Works for painsled (both iOS and desktop)

        Also copies the string to the clipboard (handy!)

        """

        stri = self.summary(separator=separator) + \
            self.intervalstats_painsled(separator=separator)

        return stri

    def plotcp(self):
        if self.empty:
            return None
        cumdist = self.df['cum_dist']
        elapsedtime = self.df[' ElapsedTime (sec)']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Duration')
        ax.set_ylabel('Power')

        delta = []
        cpvalue = []

        for i in range(len(cumdist) - 1):
            resdist = cumdist.iloc[i + 1:] - cumdist.iloc[i]  # ix -> iloc
            restime = elapsedtime.iloc[i + 1:] - elapsedtime[i] # ix -> iloc
            velo = resdist / restime
            power = 2.8 * velo**3
            power.name = 'Power'
            restime.name = 'restime'
            df = pd.concat([restime, power], axis=1).reset_index()
            maxrow = df.loc[df['Power'].idxmax()]
            delta.append(maxrow['restime'])
            cpvalue.append(maxrow['Power'])

        ax.scatter(delta, cpvalue)
        plt.show()

    def getcp(self):
        if self.empty:
            return None

        cumdist = self.df['cum_dist']
        elapsedtime = self.df[' ElapsedTime (sec)']

        delta = []
        dist = []
        cpvalue = []

        for i in range(len(cumdist) - 1):
            resdist = cumdist.iloc[i + 1:] - cumdist.iloc[i] # ix -> iloc
            restime = elapsedtime.iloc[i + 1:] - elapsedtime[i] # ix -> iloc
            velo = resdist / restime
            power = 2.8 * velo**3
            power.name = 'Power'
            restime.name = 'restime'
            resdist.name = 'resdist'
            df = pd.concat([restime, resdist, power], axis=1).reset_index()
            maxrow = df.loc[df['Power'].idxmax()]
            delta.append(maxrow['restime'])
            cpvalue.append(maxrow['Power'])
            dist.append(maxrow['resdist'])

        delta = pd.Series(delta, name='Delta')
        cpvalue = pd.Series(cpvalue, name='CP')
        dist = pd.Series(dist, name='Distance')

        return pd.concat([delta, cpvalue, dist], axis=1).reset_index()

    def plototwergpower(self):
        if self.empty:
            return None

        df = self.df
        pe = df['equivergpower']
        pw = df[' Power (watts)']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(pe, pw)
        ax.set_xlabel('Erg Power (W)')
        ax.set_ylabel('OTW Power (W)')

        plt.show()

    def plotmeters_erg(self):
        """ Creates two images containing interesting plots

        x-axis is distance

        Used with painsled (erg) data


        """
        if self.empty:
            return None

        df = self.df


        fig1 = plt.figure(figsize=(12, 10))

        # First panel, hr
        # replaced ix with loc below
        ax1 = fig1.add_subplot(4, 1, 1)
        make_hr_bars(ax1,self,df)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        make_pace_plot(ax2,self,df,mode=['distance','ote'])

        ax3 = fig1.add_subplot(4, 1, 3)
        make_spm_plot(ax3,self,df,mode=['distance'])

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        make_power_plot(ax4,self,df,mode=['distance'])

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        fig2 = plt.figure(figsize=(12, 10))
        fig_title = "Input File:  " + self.readfilename + " --- Stroke Metrics"

        # Top plot is pace
        ax5 = fig2.add_subplot(4, 1, 1)
        make_pace_plot(ax5,self,df,mode=['distance'])

        # next we plot the drive length
        ax6 = fig2.add_subplot(4, 1, 2)
        make_drivelength_plot(ax6,self,df,mode=['distance'])

        # next we plot the drive time and recovery time
        ax7 = fig2.add_subplot(4, 1, 3)
        make_drivetime_plot(ax7,self,df,mode=['distance'])

        # Peak and average force
        ax8 = fig2.add_subplot(4, 1, 4)
        make_force_plot(ax8,self,df,mode=['distance'])

        plt.subplots_adjust(hspace=0)
        fig2.subplots_adjust(hspace=0)

        plt.show()


    def plottime_erg(self):
        """ Creates two images containing interesting plots

        x-axis is time

        Used with painsled (erg) data


        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]


        fig1 = plt.figure(figsize=(12, 10))

        # First panel, hr
        ax1 = fig1.add_subplot(4, 1, 1)
        make_hr_bars(ax1,self,df,mode=['time'])

        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        make_pace_plot(ax2,self,df,mode=['time','ote'])

        # Third Panel, rate
        ax3 = fig1.add_subplot(4, 1, 3)
        make_spm_plot(ax3,self,df,mode=['time'])

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        make_power_plot(ax4,self,df,mode=['time'])
        fig1.subplots_adjust(hspace=0)



        # Top plot is pace
        fig2 = plt.figure(figsize=(12,10))
        fig_title = "Input File:  " + self.readfilename + " --- Stroke Metrics"


        ax5 = fig2.add_subplot(4, 1, 1)
        make_pace_plot(ax5,self,df,mode=['time'])

        # next we plot the drive length
        ax6 = fig2.add_subplot(4, 1, 2)
        make_drivelength_plot(ax6,self,df,mode=['time'])

        # next we plot the drive time and recovery time
        ax7 = fig2.add_subplot(4, 1, 3)
        make_drivetime_plot(ax7,self,df,mode=['time'])


        # Peak and average force
        ax8 = fig2.add_subplot(4, 1, 4)
        make_force_plot(ax8,self,df,mode=['time'])

        plt.subplots_adjust(hspace=0)
        fig2.subplots_adjust(hspace=0)

        plt.show()

        self.piechart()

    def get_metersplot_otw(self, title,*args,**kwargs):
        if self.empty:
            return None


        pacerange = kwargs.pop('pacerange',[])
        gridtrue = kwargs.pop('gridtrue',True)
        axis = kwargs.pop('axis','both')

        df = self.df
        fig1 = figure.Figure(figsize=(12, 10))

        # First panel, hr
        ax1 = fig1.add_subplot(3, 1, 1)
        make_hr_bars(ax1,self,df,mode=['distance','water'],gridtrue=gridtrue,axis=axis)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(3, 1, 2)
        make_pace_plot(ax2,self,df,mode=['distance','water'],pacerange=pacerange,gridtrue=gridtrue,axis=axis)

        # Third Panel, rate
        ax3 = fig1.add_subplot(3, 1, 3)
        make_spm_plot(ax3,self,df,mode=['distance','water'],gridtrue=gridtrue,axis=axis)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        return fig1

    def get_metersplot_erg2(self, title, *args, **kwargs):
        if self.empty:
            return None

        pacerange = kwargs.pop('pacerange',[])
        gridtrue = kwargs.pop('gridtrue',True)
        axis = kwargs.pop('axis','both')

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]
        end_dist = int(df.loc[df.index[-1], 'cum_dist'])
        fig2 = figure.Figure(figsize=(12, 10))
        fig_title = title
        if self.dragfactor:
            fig_title += " Drag %d" % self.dragfactor

        # Top plot is pace
        ax5 = fig2.add_subplot(4, 1, 1)
        make_pace_plot(ax5,self,df,mode=['distance'],gridtrue=gridtrue,axis=axis,pacerange=pacerange)

        # next we plot the drive length
        ax6 = fig2.add_subplot(4, 1, 2)
        make_drivelength_plot(ax6,self,df,mode=['distance'],gridtrue=gridtrue,axis=axis)

        # next we plot the drive time and recovery time
        ax7 = fig2.add_subplot(4, 1, 3)
        make_drivetime_plot(ax7,self,df,mode=['distance'],gridtrue=gridtrue,axis=axis)

        # Peak and average force
        ax8 = fig2.add_subplot(4, 1, 4)
        make_force_plot(ax8,self,df,mode=['distance'],gridtrue=gridtrue,axis=axis)

        plt.subplots_adjust(hspace=0)
        fig2.subplots_adjust(hspace=0)

        return fig2

    def get_timeplot_erg2(self, title, *args, **kwargs):
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]
        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])
        fig2 = figure.Figure(figsize=(12, 10))
        fig_title = title
        if self.dragfactor:
            fig_title += " Drag %d" % self.dragfactor

        # Top plot is pace
        # Top plot is pace
        ax5 = fig2.add_subplot(4, 1, 1)
        make_pace_plot(ax5,self,df,mode=['time'])

        # next we plot the drive length
        ax6 = fig2.add_subplot(4, 1, 2)
        make_drivelength_plot(ax6,self,df,mode=['time'])

        # next we plot the drive time and recovery time
        ax7 = fig2.add_subplot(4, 1, 3)
        make_drivetime_plot(ax7,self,df,mode=['time'])

        # Peak and average force
        ax8 = fig2.add_subplot(4, 1, 4)
        make_force_plot(ax8,self,df,mode=['time'])

        plt.subplots_adjust(hspace=0)
        fig2.subplots_adjust(hspace=0)

        return fig2

    def get_timeplot_otw(self, title, *args, **kwargs):
        if self.empty:
            return None

        pacerange = kwargs.pop('pacerange',[])
        gridtrue = kwargs.pop('gridtrue',True)
        axis = kwargs.pop('axis','both')

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]


        fig1 = figure.Figure(figsize=(12, 10))

        fig_title = title

        # First panel, hr
        ax1 = fig1.add_subplot(3, 1, 1)
        make_hr_bars(ax1,self,df,mode=['time','water'],title=fig_title,gridtrue=gridtrue,axis=axis)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(3, 1, 2)
        make_pace_plot(ax2,self,df,mode=['time','water'],gridtrue=gridtrue,axis=axis,pacerange=pacerange)

        # Third Panel, rate
        ax3 = fig1.add_subplot(3, 1, 3)
        make_spm_plot(ax3,self,df,mode=['time','water','last'],gridtrue=gridtrue,axis=axis)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        return fig1

    def get_pacehrplot(self, title, *args, **kwargs):
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

        t = df.loc[:, ' ElapsedTime (sec)']
        p = df.loc[:, ' Stroke500mPace (sec/500m)']
        hr = df.loc[:, ' HRCur (bpm)']
        end_time = int(df.loc[df.index[-1], ' ElapsedTime (sec)'])

        fig, ax1 = plt.subplots(figsize=(5, 4))

        ax1.plot(t, p, 'b-')
        ax1.set_xlabel('Time (h:m)')
        ax1.set_ylabel('(/500)')

        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=[85, 240], quantiles=[0, .9])
        plt.axis([0, end_time, yrange[1], yrange[0]])

        ax1.set_xticks(list(range(1000, end_time, 1000)))
        if end_time < 300:
            ax1.set_xticks(list(range(60, end_time, 60)))
        ax1.set_yticks(list(range(185, 90, -10)))
        ax1.set_title(title)
        plt.grid(True)
        majorFormatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        timeTickFormatter = NullFormatter()

        ax1.yaxis.set_major_formatter(majorFormatter)

        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2 = ax1.twinx()
        ax2.plot(t, hr, 'r-')
        ax2.set_ylabel('Heart Rate', color='r')
        majorTimeFormatter = FuncFormatter(format_time_tick)
        majorLocator = (15 * 60)
        ax2.xaxis.set_major_formatter(majorTimeFormatter)
        ax2.patch.set_alpha(0.0)
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        plt.subplots_adjust(hspace=0)
        #fig.subplots_adjust(hspace=0)

        return fig

    def bokehpaceplot(self):
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

        # time increments for bar chart
        time_increments = df.loc[:, 'TimeStamp (sec)'].diff()
        time_increments[self.index[0]] = time_increments[self.index[1]]
        time_increments = 0.5 * (abs(time_increments) + (time_increments))

        end_dist = int(df.loc[df.index[-1], 'cum_dist'])
        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])

        t = df.loc[:, ' ElapsedTime (sec)']
        p = df.loc[:, ' Stroke500mPace (sec/500m)']
        hr = df.loc[:, ' HRCur (bpm)']

        return 1

    def get_paceplot(self, title, *args, **kwargs):
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

        # time increments for bar chart
        time_increments = df.loc[:, 'TimeStamp (sec)'].diff()
        time_increments[self.index[0]] = time_increments[self.index[1]]
        time_increments = 0.5 * (abs(time_increments) + (time_increments))

        end_dist = int(df.loc[df.index[-1], 'cum_dist'])
        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])

        t = df.loc[:, ' ElapsedTime (sec)']
        p = df.loc[:, ' Stroke500mPace (sec/500m)']
        hr = df.loc[:, ' HRCur (bpm)']

        fig, ax1 = plt.subplots()
        ax1.plot(t, p, 'b-')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Pace (/500)')

        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=[85, 240], quantiles=[0, 0.9])
        plt.axis([0, end_time, yrange[1], yrange[0]])

        ax1.set_xticks(list(range(1000, end_time, 1000)))
        if end_time < 300:
            ax1.set_xticks(list(range(60, end_time, 60)))
        ax1.set_yticks(list(range(185, 90, -10)))
        ax1.set_title(title)
        grid(True)
        majorFormatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        timeTickFormatter = NullFormatter()

        ax1.yaxis.set_major_formatter(majorFormatter)
        majorTimeFormatter = FuncFormatter(format_time_tick)
        majorLocator = (15 * 60)
        ax1.xaxis.set_major_formatter(majorTimeFormatter)

        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2 = ax1.twinx()
        ax2.plot(t, hr, 'r-')
        ax2.set_ylabel('Heart Rate', color='r')

        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        return fig

    def get_metersplot_erg(self, title, *args, **kwargs):
        if self.empty:
            return None

        df = self.df

        axis = kwargs.pop('axis','both')
        gridtrue = kwargs.pop('gridtrue',True)
        pacerange = kwargs.pop('pacerange',[])

        fig1 = figure.Figure(figsize=(12, 10))

        # First panel, hr
        ax1 = fig1.add_subplot(4, 1, 1)
        make_hr_bars(ax1,self,df,axis=axis,gridtrue=gridtrue)

        grid(True)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        make_pace_plot(ax2,self,df,axis=axis,gridtrue=gridtrue,pacerange=pacerange)

        # Third Panel, rate
        ax3 = fig1.add_subplot(4, 1, 3)
        make_spm_plot(ax3,self,df,mode=['distance'],axis=axis,gridtrue=gridtrue)

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        make_power_plot(ax4,self,df,axis=axis,gridtrue=gridtrue)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        return(fig1)

    def get_metersplot_otwempower(self, title, *args, **kwargs):
        if self.empty:
            return None

        axis = kwargs.pop('axis','both')
        gridtrue = kwargs.pop('gridtrue',True)
        pacerange = kwargs.pop('pacerange',[])

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]


        fig1 = figure.Figure(figsize=(12, 10))

        # First panel, hr
        ax1 = fig1.add_subplot(4, 1, 1)
        make_hr_bars(ax1,self,df,axis=axis,gridtrue=gridtrue)


        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        make_pace_plot(ax2,self,df,mode=['distance','otw'],axis=axis,gridtrue=gridtrue,
                       pacerange=pacerange)

        # Third Panel, rate
        ax3 = fig1.add_subplot(4, 1, 3)
        make_spm_plot(ax3,self,df,axis=axis,gridtrue=gridtrue)

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        make_power_plot(ax4,self,df,axis=axis,gridtrue=gridtrue)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        return(fig1)

    def get_metersplot_otwpower(self, title, *args, **kwargs):
        if self.empty:
            return None

        axis = kwargs.pop('axis','both')
        gridtrue = kwargs.pop('gridtrue',True)
        pacerange = kwargs.pop('pacerange',[])

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]


        fig1 = figure.Figure(figsize=(12, 10))


        # First panel, hr
        ax1 = fig1.add_subplot(4, 1, 1)
        make_hr_bars(ax1,self,df,axis=axis,gridtrue=gridtrue)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        make_pace_plot(ax2,self,df,mode=['distance','otw'],axis=axis,gridtrue=gridtrue,
                       pacerange=pacerange)

        # Third Panel, rate
        ax3 = fig1.add_subplot(4, 1, 3)
        make_spm_plot(ax3,self,df,mode=['distance','otw'],axis=axis,gridtrue=gridtrue)

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        make_power_plot(ax3,self,df,mode=['distance','otw'],axis=axis,gridtrue=gridtrue)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        return(fig1)

    def get_timeplot_erg(self, title,*args,**kwargs):
        if self.empty:
            return None

        axis = kwargs.pop('axis','both')
        gridtrue = kwargs.pop('gridtrue',False)
        pacerange = kwargs.pop('pacerange',[])


        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

        fig1 = figure.Figure(figsize=(12, 10))


        # First panel, hr
        ax1 = fig1.add_subplot(4, 1, 1)
        make_hr_bars(ax1,self,df,mode=['time'],axis=axis,gridtrue=gridtrue)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        make_pace_plot(ax2,self,df,mode=['time','ote'],axis=axis,gridtrue=gridtrue,
                       pacerange=pacerange)

        # Third Panel, rate
        ax3 = fig1.add_subplot(4, 1, 3)
        make_spm_plot(ax3,self,df,mode=['time'],axis=axis,gridtrue=gridtrue)

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        make_power_plot(ax4,self,df,mode=['time'],axis=axis,gridtrue=gridtrue)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        return(fig1)

    def get_timeplot_otwempower(self, title, *args, **kwargs):
        if self.empty:
            return None

        axis = kwargs.pop('axis','both')
        gridtrue = kwargs.pop('gridtrue',True)
        pacerange = kwargs.pop('pacerange',[])

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]


        fig1 = figure.Figure(figsize=(12, 10))

        # First panel, hr
        ax1 = fig1.add_subplot(4, 1, 1)
        make_hr_bars(ax1,self,df,mode=['time','otw'],axis=axis,gridtrue=gridtrue)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        make_pace_plot(ax2,self,df,mode=['time','otw'],axis=axis,gridtrue=gridtrue,
                       pacerange=pacerange)

        ax3 = fig1.add_subplot(4, 1, 3)
        make_spm_plot(ax3,self,df,mode=['time','otw'],axis=axis,gridtrue=gridtrue)

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        make_power_plot(ax4,self,df,mode=['time','otw'],axis=axis,gridtrue=gridtrue)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        return(fig1)

    def get_time_otwpower(self, title, *args, **kwargs):
        if self.empty:
            return None

        axis = kwargs.pop('axis','both')
        gridtrue = kwargs.pop('gridtrue',True)
        pacerange = kwargs.pop('pacerange',[])

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]
        # calculate erg power

        try:
            nowindpace = df.loc[:, 'nowindpace']
        except KeyError:
            nowindpace = df[' Stroke500mPace (sec/500m)']
            df['nowindpace'] = nowindpace
        try:
            equivergpower = df.loc[:, 'equivergpower']
        except KeyError:
            equivergpower = 0 * df[' Stroke500mPace (sec/500m)'] + 50.
            df['equivergpower'] = equivergpower

        ergvelo = (equivergpower / 2.8)**(1. / 3.)

        ergpace = 500. / ergvelo
        ergpace[ergpace == np.inf] = 240.


        fig1 = figure.Figure(figsize=(12, 10))

        fig_title = title

        # First panel, hr
        ax1 = fig1.add_subplot(4, 1, 1)
        make_hr_bars(ax1,self,df,mode=['time','otw'],title=fig_title,axis=axis,gridtrue=gridtrue)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        make_pace_plot(ax2,self,df,mode=['time','otw','wind'],axis=axis,gridtrue=gridtrue,
                       pacerange=pacerange)

        # Third Panel, rate
        ax3 = fig1.add_subplot(4, 1, 3)
        make_spm_plot(ax3,self,df,mode=['time','otw'],axis=axis,gridtrue=gridtrue)

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        make_power_plot(ax4,self,df,mode=['time','otw'],axis=axis,gridtrue=gridtrue)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        return fig1

    def plottime_otwpower(self):
        """ Creates two images containing interesting plots

        x-axis is time

        Used with painsled (erg) data


        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

        # calculate erg power
        # pp=df['equivergpower']
        # ergvelo=(pp/2.8)**(1./3.)
        # relergpace=500./ergvelo

        # time increments for bar chart
        time_increments = df.loc[:, ' ElapsedTime (sec)'].diff()
        time_increments[self.index[0]] = time_increments[self.index[1]]
        time_increments = 0.5 * (abs(time_increments) + (time_increments))

        fig1 = plt.figure(figsize=(12, 10))

        fig_title = "Input File:  " + self.readfilename + " --- HR / Pace / Rate "

        # First panel, hr
        ax1 = fig1.add_subplot(4, 1, 1)
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_ut2'],
                width=time_increments,
                color='gray', ec='gray')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_ut1'],
                width=time_increments,
                color='y', ec='y')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_at'],
                width=time_increments,
                color='g', ec='g')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_tr'],
                width=time_increments,
                color='blue', ec='blue')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_an'],
                width=time_increments,
                color='violet', ec='violet')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_max'],
                width=time_increments,
                color='r', ec='r')

        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_ut2'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_ut1'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_at'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_tr'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_an'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_max'], color='k')
        ax1.text(5, self.rwr.ut2 + 1.5, self.rwr.hrzones[1], size=8)
        ax1.text(5, self.rwr.ut1 + 1.5, self.rwr.hrzones[2], size=8)
        ax1.text(5, self.rwr.at + 1.5, self.rwr.hrzones[3], size=8)
        ax1.text(5, self.rwr.tr + 1.5, self.rwr.hrzones[4], size=8)
        ax1.text(5, self.rwr.an + 1.5, self.rwr.hrzones[5], size=8)
        ax1.text(5, self.rwr.max + 1.5, self.rwr.hrzones[6], size=8)

        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])

        ax1.axis([0, end_time, 100, 1.1 * self.rwr.max])
        ax1.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax1.set_xticks(list(range(60, end_time, 60)))
        ax1.set_ylabel('BPM')
        ax1.set_yticks(list(range(110, 200, 10)))
        ax1.set_title(fig_title)
        timeTickFormatter = NullFormatter()
        ax1.xaxis.set_major_formatter(timeTickFormatter)

        grid(True)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(4, 1, 2)
        ax2.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' Stroke500mPace (sec/500m)'])

        try:
            ax2.plot(df.loc[:, 'TimeStamp (sec)'],
                     df.loc[:, 'nowindpace'])
        except KeyError:
            pass

        #       ax2.plot(df.loc[:,'TimeStamp (sec)'],
        #        ergpace)

        ax2.legend(['Pace', 'Wind corrected pace'],
                   prop={'size': 10}, loc=0)

        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])

        try:
            s = np.concatenate((df.loc[:, ' Stroke500mPace (sec/500m)'].values,
                                df.loc[:, 'nowindpace'].values))
        except KeyError:
            s = df.loc[:, ' Stroke500mPace (sec/500m)'].values

        yrange = y_axis_range(s, ultimate=[90, 240], quantiles=[0.0, 0.9])

        ax2.axis([0, end_time, yrange[1], yrange[0]])
        ax2.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax2.set_xticks(list(range(60, end_time, 60)))
        ax2.set_ylabel('(/500)')
#       ax2.set_yticks(range(145,90,-5))
        # ax2.set_title('Pace')
        grid(True)
        majorFormatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        ax2.xaxis.set_major_formatter(timeTickFormatter)
        ax2.yaxis.set_major_formatter(majorFormatter)

        # Third Panel, rate
        ax3 = fig1.add_subplot(4, 1, 3)
        ax3.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' Cadence (stokes/min)'])
#       rate_ewma=pd.ewma
        ax3.axis([0, end_time, 14, 40])
        ax3.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax3.set_xticks(list(range(60, end_time, 60)))
        ax3.set_xlabel('Time (sec)')
        ax3.set_ylabel('SPM')
        ax3.set_yticks(list(range(16, 40, 2)))
        # ax3.set_title('Rate')
        ax3.xaxis.set_major_formatter(timeTickFormatter)
        grid(True)

        # Fourth Panel, watts
        ax4 = fig1.add_subplot(4, 1, 4)
        ax4.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, ' Power (watts)'])
        # ax4.plot(df.loc[:,'TimeStamp (sec)'],df.loc[:,'equivergpower'])
        ax4.legend(['Power'], prop={'size': 10})
        yrange = y_axis_range(df.loc[:, ' Power (watts)'],
                              ultimate=[0, 555], miny=0)
        ax4.axis([0, end_time, yrange[0], yrange[1]])
        ax4.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax4.set_xticks(list(range(60, end_time, 60)))
        ax4.set_xlabel('Time (h:m)')
        ax4.set_ylabel('Watts')
#       ax4.set_yticks(range(150,450,50))
        # ax4.set_title('Power')
        grid(True)
        majorTimeFormatter = FuncFormatter(format_time_tick)
        majorLocator = (15 * 60)
        ax4.xaxis.set_major_formatter(majorTimeFormatter)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        fig2 = plt.figure(figsize=(12, 10))
        fig_title = "Input File:  " + self.readfilename + " --- Stroke Metrics"

        # Top plot is pace
        ax5 = fig2.add_subplot(4, 1, 1)
        ax5.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' Stroke500mPace (sec/500m)'])

        try:
            ax5.plot(df.loc[:, 'TimeStamp (sec)'],
                     df.loc[:, 'nowindpace'])
        except KeyError:
            pass

        # ax5.plot(df.loc[:,'TimeStamp (sec)'],
        #        ergpace)

        ax5.legend(['Pace', 'Wind corrected pace'],
                   prop={'size': 10}, loc=0)

        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])

        try:
            s = np.concatenate((df.loc[:, ' Stroke500mPace (sec/500m)'].values,
                                df.loc[:, 'nowindpace'].values))
        except KeyError:
            s = df.loc[:, ' Stroke500mPace (sec/500m)'].values

        yrange = y_axis_range(s, ultimate=[90, 240], quantiles=[0.0, 0.9])

        ax5.axis([0, end_time, yrange[1], yrange[0]])
        ax5.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax5.set_xticks(list(range(60, end_time, 60)))
        ax5.set_ylabel('(/500)')
#       ax5.set_yticks(range(145,90,-5))
        grid(True)
        ax5.set_title(fig_title)
        majorFormatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        ax5.xaxis.set_major_formatter(timeTickFormatter)
        ax5.yaxis.set_major_formatter(majorFormatter)

        # next we plot the drive length
        ax6 = fig2.add_subplot(4, 1, 2)
        ax6.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' DriveLength (meters)'])
        yrange = y_axis_range(df.loc[:, ' DriveLength (meters)'],
                              ultimate=[1.0, 15])
        ax6.axis([0, end_time, yrange[0], yrange[1]])
        ax6.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax6.set_xticks(list(range(60, end_time, 60)))
        ax6.set_xlabel('Time (sec)')
        ax6.set_ylabel('Drive Len(m)')
#       ax6.set_yticks(np.arange(1.35,1.6,0.05))
        ax6.xaxis.set_major_formatter(timeTickFormatter)
        grid(True)

        # next we plot the drive time and recovery time
        ax7 = fig2.add_subplot(4, 1, 3)
        ax7.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' DriveTime (ms)'] / 1000.)
        ax7.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' StrokeRecoveryTime (ms)'] / 1000.)
        s = np.concatenate((df.loc[:, ' DriveTime (ms)'].values / 1000.,
                            df.loc[:, ' StrokeRecoveryTime (ms)'].values / 1000.))
        yrange = y_axis_range(s, ultimate=[0.5, 4])

        ax7.axis([0, end_time, yrange[0], yrange[1]])
        ax7.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax7.set_xticks(list(range(60, end_time, 60)))
        ax7.set_xlabel('Time (sec)')
        ax7.set_ylabel('Drv / Rcv Time (s)')
#       ax7.set_yticks(np.arange(0.2,3.0,0.2))
        ax7.xaxis.set_major_formatter(timeTickFormatter)
        grid(True)

        # Peak and average force
        ax8 = fig2.add_subplot(4, 1, 4)
        ax8.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' AverageDriveForce (lbs)'] * lbstoN)
        ax8.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' PeakDriveForce (lbs)'] * lbstoN)
        s = np.concatenate((df.loc[:, ' AverageDriveForce (lbs)'].values * lbstoN,
                            df.loc[:, ' PeakDriveForce (lbs)'].values * lbstoN))
        yrange = y_axis_range(s, ultimate=[0, 1000])

        ax8.axis([0, end_time, yrange[0], yrange[1]])
        ax8.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax8.set_xticks(list(range(60, end_time, 60)))
        ax8.set_xlabel('Time (h:m)')
        ax8.set_ylabel('Force (N)')
#       ax8.set_yticks(range(25,300,25))
        # ax4.set_title('Power')
        grid(True)
        majorTimeFormatter = FuncFormatter(format_time_tick)
        majorLocator = (15 * 60)
        ax8.xaxis.set_major_formatter(majorTimeFormatter)

        plt.subplots_adjust(hspace=0)
        fig2.subplots_adjust(hspace=0)

        plt.show()

        self.piechart()


    def plottime_hr(self):
        """ Creates a HR vs time plot

        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

        fig1 = plt.figure(figsize=(12, 10))
        fig_title = "Input File:  " + self.readfilename + " --- HR "

        # First panel, hr
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.bar(df.loc[:, 'TimeStamp (sec)'],
                df.loc[:, 'hr_ut2'], color='gray', ec='gray')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'],
                df.loc[:, 'hr_ut1'], color='y', ec='y')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'],
                df.loc[:, 'hr_at'], color='g', ec='g')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'],
                df.loc[:, 'hr_tr'], color='blue', ec='blue')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'],
                df.loc[:, 'hr_an'], color='violet', ec='violet')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'],
                df.loc[:, 'hr_max'], color='r', ec='r')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_ut2'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_ut1'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_at'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_tr'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_an'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_max'], color='k')
        ax1.text(5, self.rwr.ut2 + 1.5, self.rwr.hrzones[1], size=8)
        ax1.text(5, self.rwr.ut1 + 1.5, self.rwr.hrzones[2], size=8)
        ax1.text(5, self.rwr.at + 1.5, self.rwr.hrzones[3], size=8)
        ax1.text(5, self.rwr.tr + 1.5, self.rwr.hrzones[4], size=8)
        ax1.text(5, self.rwr.an + 1.5, self.rwr.hrzones[5], size=8)
        ax1.text(5, self.rwr.max + 1.5, self.rwr.hrzones[6], size=8)

        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])
        ax1.axis([0, end_time, 100, 1.1 * self.rwr.max])
        ax1.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax1.set_xticks(list(range(60, end_time, 60)))
        ax1.set_ylabel('BPM')
        ax1.set_yticks(list(range(110, 190, 10)))
        ax1.set_title(fig_title)
        timeTickFormatter = NullFormatter()
        ax1.xaxis.set_major_formatter(timeTickFormatter)

        grid(True)
        plt.show()

    def plotmeters_otw(self):
        """ Creates two images containing interesting plots

        x-axis is distance

        Used with OTW data (no Power plot)


        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

        # distance increments for bar chart
        dist_increments = -df.loc[:, 'cum_dist'].diff()
        dist_increments[0] = dist_increments[1]
#       dist_increments=abs(dist_increments)+dist_increments

        fig1 = plt.figure(figsize=(12, 10))
        fig_title = "Input File:  " + self.readfilename + " --- HR / Pace / Rate / Power"

        # First panel, hr
        ax1 = fig1.add_subplot(3, 1, 1)
        ax1.bar(df.loc[:, 'cum_dist'], df.loc[:, 'hr_ut2'],
                width=dist_increments, align='edge',
                color='gray', ec='gray')
        ax1.bar(df.loc[:, 'cum_dist'], df.loc[:, 'hr_ut1'],
                width=dist_increments, align='edge',
                color='y', ec='y')
        ax1.bar(df.loc[:, 'cum_dist'], df.loc[:, 'hr_at'],
                width=dist_increments, align='edge',
                color='g', ec='g')
        ax1.bar(df.loc[:, 'cum_dist'], df.loc[:, 'hr_tr'],
                width=dist_increments, align='edge',
                color='blue', ec='blue')
        ax1.bar(df.loc[:, 'cum_dist'], df.loc[:, 'hr_an'],
                width=dist_increments, align='edge',
                color='violet', ec='violet')
        ax1.bar(df.loc[:, 'cum_dist'], df.loc[:, 'hr_max'],
                width=dist_increments, align='edge',
                color='r', ec='r')

        ax1.plot(df.loc[:, 'cum_dist'], df.loc[:, 'lim_ut2'], color='k')
        ax1.plot(df.loc[:, 'cum_dist'], df.loc[:, 'lim_ut1'], color='k')
        ax1.plot(df.loc[:, 'cum_dist'], df.loc[:, 'lim_at'], color='k')
        ax1.plot(df.loc[:, 'cum_dist'], df.loc[:, 'lim_tr'], color='k')
        ax1.plot(df.loc[:, 'cum_dist'], df.loc[:, 'lim_an'], color='k')
        ax1.plot(df.loc[:, 'cum_dist'], df.loc[:, 'lim_max'], color='k')

        ax1.text(5, self.rwr.ut2 + 1.5, self.rwr.hrzones[1], size=8)
        ax1.text(5, self.rwr.ut1 + 1.5, self.rwr.hrzones[2], size=8)
        ax1.text(5, self.rwr.at + 1.5, self.rwr.hrzones[3], size=8)
        ax1.text(5, self.rwr.tr + 1.5, self.rwr.hrzones[4], size=8)
        ax1.text(5, self.rwr.an + 1.5, self.rwr.hrzones[5], size=8)
        ax1.text(5, self.rwr.max + 1.5, self.rwr.hrzones[6], size=8)

        end_dist = int(df.loc[df.index[-1], 'cum_dist'])

        ax1.axis([0, end_dist, 100, 1.1 * self.rwr.max])
        ax1.set_xticks(list(range(1000, end_dist, 1000)))
        if end_dist < 1000:
            ax1.set_xticks(list(range(100, end_dist, 100)))
        ax1.set_ylabel('BPM')
        ax1.set_yticks(list(range(110, 200, 10)))
        ax1.set_title(fig_title)

        grid(True)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(3, 1, 2)
        ax2.plot(df.loc[:, 'cum_dist'], df.loc[:, ' Stroke500mPace (sec/500m)'])
        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=[85, 240], quantiles=[0.0, 0.9])

        ax2.axis([0, end_dist, yrange[1], yrange[0]])
        ax2.set_xticks(list(range(1000, end_dist, 1000)))
        if end_dist < 1000:
            ax2.set_xticks(list(range(100, end_dist, 100)))
        ax2.set_ylabel('(/500m)')
#       ax2.set_yticks(range(175,95,-10))
        grid(True)
        majorTickFormatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        ax2.yaxis.set_major_formatter(majorTickFormatter)

        # Third Panel, rate
        ax3 = fig1.add_subplot(3, 1, 3)
        ax3.plot(df.loc[:, 'cum_dist'], df.loc[:, ' Cadence (stokes/min)'])
        ax3.axis([0, end_dist, 14, 40])
        ax3.set_xticks(list(range(1000, end_dist, 1000)))
        if end_dist < 1000:
            ax3.set_xticks(list(range(100, end_dist, 100)))
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('SPM')
        ax3.set_yticks(list(range(16, 40, 2)))

        grid(True)

        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        fig2 = plt.figure(figsize=(12, 10))
        fig_title = "Input File:  " + self.readfilename + " --- Stroke Metrics"

        # Top plot is pace
        ax5 = fig2.add_subplot(2, 1, 1)
        ax5.plot(df.loc[:, 'cum_dist'], df.loc[:, ' Stroke500mPace (sec/500m)'])
        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=[85, 240], quantiles=[0.0, 0.9])
        ax5.axis([0, end_dist, yrange[1], yrange[0]])
        ax5.set_xticks(list(range(1000, end_dist, 1000)))
        if end_dist < 1000:
            ax5.set_xticks(list(range(100, end_dist, 100)))
        ax5.set_ylabel('(/500)')
#       ax5.set_yticks(range(175,95,-10))
        grid(True)
        ax5.set_title(fig_title)
        majorFormatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        ax5.yaxis.set_major_formatter(majorFormatter)

        # next we plot the stroke distance
        ax6 = fig2.add_subplot(2, 1, 2)
        ax6.plot(df.loc[:, 'cum_dist'], df.loc[:, ' StrokeDistance (meters)'])
        yrange = y_axis_range(df.loc[:, ' StrokeDistance (meters)'],
                              ultimate=[5, 15])
        ax6.axis([0, end_dist, yrange[0], yrange[1]])
        ax6.set_xlabel('Distance (m)')
        ax6.set_xticks(list(range(1000, end_dist, 1000)))
        if end_dist < 1000:
            ax6.set_xticks(list(range(100, end_dist, 100)))
        ax6.set_ylabel('Stroke Distance (m)')
#       ax6.set_yticks(np.arange(5.5,11.5,0.5))
        grid(True)

        plt.subplots_adjust(hspace=0)
        fig2.subplots_adjust(hspace=0)

        plt.show()

    def plottime_otw(self):
        """ Creates two images containing interesting plots

        x-axis is time

        Used with OTW data (no Power plot)


        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

        # time increments for bar chart
        time_increments = df.loc[:, ' ElapsedTime (sec)'].diff()
        time_increments[self.index[0]] = time_increments[self.index[1]]
        time_increments = 0.5 * (abs(time_increments) + (time_increments))

        fig1 = plt.figure(figsize=(12, 10))
        fig_title = "Input File:  " + self.readfilename + " --- HR / Pace / Rate "

        # First panel, hr
        ax1 = fig1.add_subplot(3, 1, 1)
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_ut2'],
                width=time_increments,
                color='gray', ec='gray')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_ut1'],
                width=time_increments,
                color='y', ec='y')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_at'],
                width=time_increments,
                color='g', ec='g')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_tr'],
                width=time_increments,
                color='blue', ec='blue')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_an'],
                width=time_increments,
                color='violet', ec='violet')
        ax1.bar(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'hr_max'],
                width=time_increments,
                color='r', ec='r')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_ut2'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_ut1'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_at'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_tr'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_an'], color='k')
        ax1.plot(df.loc[:, 'TimeStamp (sec)'], df.loc[:, 'lim_max'], color='k')
        ax1.text(5, self.rwr.ut2 + 1.5, self.rwr.hrzones[1], size=8)
        ax1.text(5, self.rwr.ut1 + 1.5, self.rwr.hrzones[2], size=8)
        ax1.text(5, self.rwr.at + 1.5, self.rwr.hrzones[3], size=8)
        ax1.text(5, self.rwr.tr + 1.5, self.rwr.hrzones[4], size=8)
        ax1.text(5, self.rwr.an + 1.5, self.rwr.hrzones[5], size=8)
        ax1.text(5, self.rwr.max + 1.5, self.rwr.hrzones[6], size=8)

        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])
        ax1.axis([0, end_time, 100, 1.1 * self.rwr.max])
        ax1.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax1.set_xticks(list(range(60, end_time, 60)))
        ax1.set_ylabel('BPM')
        ax1.set_yticks(list(range(110, 190, 10)))
        ax1.set_title(fig_title)
        timeTickFormatter = NullFormatter()
        ax1.xaxis.set_major_formatter(timeTickFormatter)

        grid(True)

        # Second Panel, Pace
        ax2 = fig1.add_subplot(3, 1, 2)
        ax2.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' Stroke500mPace (sec/500m)'])
        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])
        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=[85, 240], quantiles=[0.0, 0.9])
        ax2.axis([0, end_time, yrange[1], yrange[0]])
        ax2.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax2.set_xticks(list(range(60, end_time, 60)))
        ax2.set_ylabel('(/500m)')
#       ax2.set_yticks(range(175,90,-5))
        # ax2.set_title('Pace')
        ax2.grid(True,which='major',axis='y')
        majorFormatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        ax2.xaxis.set_major_formatter(timeTickFormatter)
        ax2.yaxis.set_major_formatter(majorFormatter)

        # Third Panel, rate
        ax3 = fig1.add_subplot(3, 1, 3)
        ax3.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' Cadence (stokes/min)'])
#       rate_ewma=pd.ewma(df,span=20)
#       ax3.plot(rate_ewma.loc[:,'TimeStamp (sec)'],
#                rate_ewma.loc[:,' Cadence (stokes/min)'])
        ax3.axis([0, end_time, 14, 40])
        ax3.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax3.set_xticks(list(range(60, end_time, 60)))
        ax3.set_xlabel('Time (sec)')
        ax3.set_ylabel('SPM')
        ax3.set_yticks(list(range(16, 40, 2)))
        # ax3.set_title('Rate')
        ax3.xaxis.set_major_formatter(timeTickFormatter)
        grid(True)

        majorTimeFormatter = FuncFormatter(format_time_tick)
        majorLocator = (15 * 60)
        ax3.set_xlabel('Time (h:m)')
        ax3.xaxis.set_major_formatter(majorTimeFormatter)
        plt.subplots_adjust(hspace=0)
        fig1.subplots_adjust(hspace=0)

        fig2 = plt.figure(figsize=(12, 10))
        fig_title = "Input File:  " + self.readfilename + " --- Stroke Metrics"

        # Top plot is pace
        ax5 = fig2.add_subplot(2, 1, 1)
        ax5.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' Stroke500mPace (sec/500m)'])
        yrange = y_axis_range(df.loc[:, ' Stroke500mPace (sec/500m)'],
                              ultimate=[85, 240], quantiles=[0.0, 0.9])
        end_time = int(df.loc[df.index[-1], 'TimeStamp (sec)'])
        ax5.axis([0, end_time, yrange[1], yrange[0]])
        ax5.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax5.set_xticks(list(range(60, end_time, 60)))
        ax5.set_ylabel('(/500m)')
#       ax5.set_yticks(range(175,90,-5))
        grid(True)
        ax5.set_title(fig_title)
        majorFormatter = FuncFormatter(format_pace_tick)
        majorLocator = (5)
        ax5.xaxis.set_major_formatter(timeTickFormatter)
        ax5.yaxis.set_major_formatter(majorFormatter)

        # next we plot the drive length
        ax6 = fig2.add_subplot(2, 1, 2)
        ax6.plot(df.loc[:, 'TimeStamp (sec)'],
                 df.loc[:, ' StrokeDistance (meters)'])
        yrange = y_axis_range(df.loc[:, ' StrokeDistance (meters)'],
                              ultimate=[5, 15])

        ax6.axis([0, end_time, yrange[0], yrange[1]])
        ax6.set_xticks(list(range(0, end_time, 300)))
        if end_time < 300:
            ax6.set_xticks(list(range(60, end_time, 60)))
        ax6.set_xlabel('Time (sec)')
        ax6.set_ylabel('Stroke Distance (m)')
#       ax6.set_yticks(np.arange(5.5,11.5,0.5))
        ax6.xaxis.set_major_formatter(timeTickFormatter)
        grid(True)

        majorTimeFormatter = FuncFormatter(format_time_tick)
        majorLocator = (15 * 60)
        ax6.set_xlabel('Time (h:m)')
        ax6.xaxis.set_major_formatter(majorTimeFormatter)
        plt.subplots_adjust(hspace=0)
        fig2.subplots_adjust(hspace=0)

        plt.show()

        self.piechart()


    def piechart(self):
        """ Figure 3 - Heart Rate Time in band.
        This is not as simple as just totalling up the
        hits for each band of HR.  Since each data point represents
        a different increment of time.  This loop scans through the
        HR data and adds that incremental time in each band

        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]
#       df.sort_values(by=' ElapsedTime (sec)',ascending=1)
        df.sort_values(by='TimeStamp (sec)', ascending=1)
        number_of_rows = self.number_of_rows

        time_increments = df.loc[:, 'TimeStamp (sec)'].diff()
        time_increments[self.index[0]] = time_increments[self.index[1]]
        time_increments = 0.5 * (abs(time_increments) + (time_increments))

        time_in_zone = np.zeros(6)
        for i in df.index:
            if df.loc[i, ' HRCur (bpm)'] <= self.rwr.ut2:
                time_in_zone[0] += time_increments[self.index[i]]
            elif df.loc[i, ' HRCur (bpm)'] <= self.rwr.ut1:
                time_in_zone[1] += time_increments[self.index[i]]
            elif df.loc[i, ' HRCur (bpm)'] <= self.rwr.at:
                time_in_zone[2] += time_increments[self.index[i]]
            elif df.loc[i, ' HRCur (bpm)'] <= self.rwr.tr:
                time_in_zone[3] += time_increments[self.index[i]]
            elif df.loc[i, ' HRCur (bpm)'] <= self.rwr.an:
                time_in_zone[4] += time_increments[self.index[i]]
            else:
                time_in_zone[5] += time_increments[self.index[i]]

        # print(time_in_zone)
        wedge_labels = list(self.rwr.hrzones[0:6])

        totaltime = time_in_zone.sum()

        perc = 100. * time_in_zone / totaltime
        cutoff = 1.0
        if len(perc[perc < cutoff]) > 1:
            cutoff = 2.0
            if len(perc[perc < cutoff]) > 1:
                cutoff = 3.0

        for i in range(len(wedge_labels)):
            minutes = int(time_in_zone[i] / 60.)
            sec = int(time_in_zone[i] - minutes * 60.)
            secstr = str(sec).zfill(2)
            s = "%d:%s" % (minutes, secstr)
            wedge_labels[i] = wedge_labels[i] + "\n" + s
            perc = 100. * time_in_zone[i] / totaltime
            if perc < cutoff:
                wedge_labels[i] = ''

        # print(wedge_labels)
        fig2 = plt.figure(figsize=(5, 5))
        fig_title = "Input File:  " + self.readfilename + " --- HR Time in Zone"
        ax9 = fig2.add_subplot(1, 1, 1)
        ax9.pie(time_in_zone,
                labels=wedge_labels,
                colors=['gray', 'gold', 'limegreen', 'dodgerblue', 'm', 'r'],
                autopct=lambda x: my_autopct(x, cutoff=cutoff + 2),
                pctdistance=0.8,
                counterclock=False,
                startangle=90.0)

        ax9.set_title(fig_title)

        plt.show()
        return 1

    def power_piechart(self):
        """ Figure 3 - Heart Rate Time in band.
        This is not as simple as just totalling up the
        hits for each band of HR.  Since each data point represents
        a different increment of time.  This loop scans through the
        HR data and adds that incremental time in each band

        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]

#       df.sort_values(by=' ElapsedTime (sec)',ascending=1)
        df.sort_values(by='TimeStamp (sec)', ascending=1)
        number_of_rows = self.number_of_rows

        time_increments = df.loc[:, 'TimeStamp (sec)'].diff()
        time_increments[self.index[0]] = time_increments[self.index[1]]
        time_increments = 0.5 * (abs(time_increments) + (time_increments))

        ut2, ut1, at, tr, an = self.rwr.ftp * \
            np.array(self.rwr.powerperc) / 100.

        time_in_zone = np.zeros(6)
        for i in df.index:
            if df.loc[i, ' Power (watts)'] <= ut2:
                time_in_zone[0] += time_increments[i]
            elif df.loc[i, ' Power (watts)'] <= ut1:
                time_in_zone[1] += time_increments[i]
            elif df.loc[i, ' Power (watts)'] <= at:
                time_in_zone[2] += time_increments[i]
            elif df.loc[i, ' Power (watts)'] <= tr:
                time_in_zone[3] += time_increments[i]
            elif df.loc[i, ' Power (watts)'] <= an:
                time_in_zone[4] += time_increments[i]
            else:
                time_in_zone[5] += time_increments[i]

        # print(time_in_zone)
        wedge_labels = list(self.rwr.powerzones)
        #['power<ut2','power ut2','power ut1','power at',
        #               'power tr','power an']

        totaltime = time_in_zone.sum()
        perc = 100. * time_in_zone / totaltime
        cutoff = 1.0
        if len(perc[perc < cutoff]) > 1:
            cutoff = 2.0
            if len(perc[perc < cutoff]) > 1:
                cutoff = 3.0

        for i in range(len(wedge_labels)):
            min = int(time_in_zone[i] / 60.)
            sec = int(time_in_zone[i] - min * 60.)
            secstr = str(sec).zfill(2)
            s = "%d:%s" % (min, secstr)
            wedge_labels[i] = wedge_labels[i] + "\n" + s
            perc = 100. * time_in_zone[i] / totaltime

            if perc < cutoff:
                wedge_labels[i] = ''

        # print(wedge_labels)
        fig2 = plt.figure(figsize=(5, 5))
        fig_title = "Input File:  " + self.readfilename + " --- Power Time in Zone"
        ax9 = fig2.add_subplot(1, 1, 1)
        ax9.pie(time_in_zone,
                labels=wedge_labels,
                colors=['gray', 'gold', 'limegreen', 'dodgerblue', 'm', 'r'],
                autopct=lambda x: my_autopct(x, cutoff=cutoff + 2),
                pctdistance=0.8,
                counterclock=False,
                startangle=90.0)

        ax9.set_title(fig_title)

        plt.show()
        return 1

    def get_power_piechart(self, title, *args, **kwargs):
        """ Figure 3 - Heart Rate Time in band.
        This is not as simple as just totalling up the
        hits for each band of HR.  Since each data point represents
        a different increment of time.  This loop scans through the
        HR data and adds that incremental time in each band

        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]
#       df.sort_values(by=' ElapsedTime (sec)',ascending=1)
        df.sort_values(by='TimeStamp (sec)', ascending=1)
        number_of_rows = self.number_of_rows

        time_increments = df.loc[:, 'TimeStamp (sec)'].diff()
        time_increments[self.index[0]] = time_increments[self.index[1]]
        time_increments = 0.5 * (abs(time_increments) + (time_increments))
        time_increments[time_increments>10] = 10.

        ut2, ut1, at, tr, an = self.rwr.ftp * \
            np.array(self.rwr.powerperc) / 100.

        time_in_zone = np.zeros(6)
        for i in df.index:
            if df.loc[i, ' Power (watts)'] <= ut2:
                time_in_zone[0] += time_increments[i]
            elif df.loc[i, ' Power (watts)'] <= ut1:
                time_in_zone[1] += time_increments[i]
            elif df.loc[i, ' Power (watts)'] <= at:
                time_in_zone[2] += time_increments[i]
            elif df.loc[i, ' Power (watts)'] <= tr:
                time_in_zone[3] += time_increments[i]
            elif df.loc[i, ' Power (watts)'] <= an:
                time_in_zone[4] += time_increments[i]
            else:
                time_in_zone[5] += time_increments[i]

        # print(time_in_zone)
        wedge_labels = list(self.rwr.powerzones)

        #['power<ut2','power ut2','power ut1','power at',
        #               'power tr','power an']

        totaltime = time_in_zone.sum()
        perc = 100. * time_in_zone / totaltime
        cutoff = 1.0
        if len(perc[perc < cutoff]) > 1:
            cutoff = 2.0
            if len(perc[perc < cutoff]) > 1:
                cutoff = 3.0

        for i in range(len(wedge_labels)):
            min = int(time_in_zone[i] / 60.)
            sec = int(time_in_zone[i] - min * 60.)
            secstr = str(sec).zfill(2)
            s = "%d:%s" % (min, secstr)
            wedge_labels[i] = wedge_labels[i] + "\n" + s
            perc = 100. * time_in_zone[i] / totaltime
            if perc < 5:
                wedge_labels[i] = ''

        # print(wedge_labels)
        fig2 = figure.Figure(figsize=(5, 5))
        fig_title = title
        ax9 = fig2.add_subplot(1, 1, 1)
        ax9.pie(time_in_zone,
                labels=wedge_labels,
                colors=['gray', 'gold', 'limegreen', 'dodgerblue', 'm', 'r'],
                autopct=lambda x: my_autopct(x, cutoff=cutoff + 2),
                pctdistance=0.8,
                counterclock=False,
                startangle=90.0)

        ax9.set_title(title)

        return fig2

    def get_piechart(self, title, *args, **kwargs):
        """ Figure 3 - Heart Rate Time in band.
        This is not as simple as just totalling up the
        hits for each band of HR.  Since each data point represents
        a different increment of time.  This loop scans through the
        HR data and adds that incremental time in each band

        """
        if self.empty:
            return None

        df = self.df
        if self.absolutetimestamps:
            df['TimeStamp (sec)'] = df['TimeStamp (sec)'] - \
                df['TimeStamp (sec)'].values[0]
        number_of_rows = self.number_of_rows

        time_increments = df.loc[:, 'TimeStamp (sec)'].diff()
        time_increments[self.index[0]] = time_increments[self.index[1]]
        time_increments = 0.5 * (abs(time_increments) + (time_increments))

        time_increments[time_increments>10] = 10.

        time_in_zone = np.zeros(6)
        for i in df.index:
            if df.loc[i, ' HRCur (bpm)'] <= self.rwr.ut2:
                time_in_zone[0] += time_increments[i]
            elif df.loc[i, ' HRCur (bpm)'] <= self.rwr.ut1:
                time_in_zone[1] += time_increments[i]
            elif df.loc[i, ' HRCur (bpm)'] <= self.rwr.at:
                time_in_zone[2] += time_increments[i]
            elif df.loc[i, ' HRCur (bpm)'] <= self.rwr.tr:
                time_in_zone[3] += time_increments[i]
            elif df.loc[i, ' HRCur (bpm)'] <= self.rwr.an:
                time_in_zone[4] += time_increments[i]
            else:
                time_in_zone[5] += time_increments[i]

        # print(time_in_zone)
        wedge_labels = list(self.rwr.hrzones[0:6])
        totaltime = time_in_zone.sum()
        perc = 100. * time_in_zone / totaltime
        cutoff = 1.0
        if len(perc[perc < cutoff]) > 1:
            cutoff = 2.0
            if len(perc[perc < cutoff]) > 1:
                cutoff = 3.0

        for i in range(len(wedge_labels)):
            min = int(time_in_zone[i] / 60.)
            sec = int(time_in_zone[i] - min * 60.)
            secstr = str(sec).zfill(2)
            s = "%d:%s" % (min, secstr)
            wedge_labels[i] = wedge_labels[i] + "\n" + s
            perc = 100. * time_in_zone[i] / totaltime
            if perc < 5:
                wedge_labels[i] = ''

        # print(wedge_labels)
#       fig2=figure.Figure(figsize=(5,5))
        fig2 = figure.Figure(figsize=(5, 5))
        fig_title = title
        ax9 = fig2.add_subplot(1, 1, 1)
        ax9.pie(time_in_zone,
                labels=wedge_labels,
                colors=['gray', 'gold', 'limegreen', 'dodgerblue', 'm', 'r'],
                autopct=lambda x: my_autopct(x, cutoff=cutoff + 2),
                pctdistance=0.8,
                counterclock=False,
                startangle=90.0)

        ax9.set_title(fig_title)

        return fig2



def dorowall(readFile="testdata", window_size=20):
    """ Used if you have CrewNerd TCX and summary CSV with the same file name

    Creates all the plots and spits out a text summary (and copies it
    to the clipboard too!)

    """

    tcxFile = readFile + ".TCX"
    csvsummary = readFile + ".CSV"
    csvoutput = readFile + "_data.CSV"

    tcx = rowingdata.TCXParser(tcxFile)
    tcx.write_csv(csvoutput, window_size=window_size)

    res = rowingdata.rowingdata(csvoutput)
    res.plotmeters_otw()

    sumdata = rowingdata.summarydata(csvsummary)
    sumdata.shortstats()

    sumdata.allstats()
