#! /usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
from six.moves import input
import math
import time
import pickle
import pylab
import numpy as np
import pandas as pd

#import scipy
import matplotlib 
from matplotlib import pyplot


from numpy import linspace,zeros,cumsum,mean
from six.moves import range

__version__ = "0.5.0"

# submodules
from .srnumerical import *
from .row_functions import *

# global parameters

# boat drag coefficient
#alfa = 3.5  # 2.95??  for skif, from Marinus van Holst

alfa = 3.06 # best fit to Kleshnev data for single 
# alfaatkinson = 3.18 # use for Atkinson
alfaatkinson = 3.4
rho_air = 1.226 # kg/m3
Cdw = 1.1 # for all boats - big approximation
#crewarea = 1.4
crewarea = 2.0
scalepower = 0.67

def main():
    return "Executing rowingphysics version %s." % __version__


def time500mtovavg(minutes,secs):
    """ Calculates velocity from pace in minutes, seconds)

    """
    seconds = 60.*minutes+secs
    vavg = 500./seconds
    return vavg

def vavgto500mtime(vavg):
    """ Calculates 500m time (minutes, seconds) from velocity

    """
    seconds = 500.0/vavg
    minutes = np.floor(seconds/60.)
    secs = seconds-60.0*minutes
   
    return [minutes,secs]

def write_obj(obj,filename):
   pickle.dump(obj,open(filename,"wb"))

def read_obj(filename):
   res = pickle.load(open(filename))
   return res


def testbladeforce(fhandle,rigging,vb,oarangle=0.01,aantal=10):
    """ iterates slip using "real" fulcrum point
    aantal = nr iterations

    """
    lin = rigging.lin
    lscull = rigging.lscull
    lout = lscull - lin
    oarangle = oarangle*np.pi/180.

    Fblade = fhandle*lin/lout 
    res = blade_force(oarangle,rigging,vb,Fblade)
    phidot = res[0]
   
    print((Fblade,180*phidot/np.pi,180.*vb*np.cos(0.01)/(np.pi*lout)))


    Fb = zeros(aantal)
    itern = list(range(aantal))
   
    for i in range(aantal):
        l2 = lout
        Fb[i] = fhandle*lin/l2
        res = blade_force(oarangle,rigging,vb,Fb[i])
        phidot = res[0]
        print((Fb[i],180.*phidot/np.pi))
        
        Fdot = fhandle + Fb

    # plot
    pyplot.clf()
    pyplot.subplot(111)
    pyplot.plot(itern, Fb,'ro',label = 'Blade Force')
    pyplot.plot(itern, Fdol,'bo',label = 'Oarlock Force')
    pylab.legend()
    pyplot.xlabel("Iteration")
    pyplot.ylabel('Force (N)')
    pyplot.show()


def plotforce(fhandle,rigging,vb,oarangle=0.01):
    """ iterates slip using "real" fulcrum point
    aantal = nr iterations
    """
   
    lin = rigging.lin
    lscull = rigging.lscull
    lout = lscull - lin
    oarangle = oarangle*np.pi/180.

    Fblade = fhandle*lin/lout 
    res = blade_force(oarangle,rigging,vb,Fblade,doplot=1)
    phidot = res[0]
   
    print((Fblade,180*phidot/np.pi,180.*vb*np.cos(0.01)/(np.pi*lout)))


def empirical(datafile,vavg,crew,rigging,tstroke,trecovery,doplot=1):
    """ Reads in empirical acceleration data to be compared with
    acceleration plot

    """
    
    lin = rigging.lin
    lscull = rigging.lscull
    lout = lscull - lin
    tempo = crew.tempo
    mc = crew.mc
    mb = rigging.mb
    Nrowers = rigging.Nrowers
    try:
        dragform = rigging.dragform
    except AttributeError:
        dragform = 1.0

    catchangle = rigging.oarangle(0)

    empdata = np.genfromtxt(datafile, delimiter = ',',skip_header=1)
    emptime = empdata[:,0]
    empdt = emptime[1]-emptime[0]
    empdtarray = gradient(emptime)
    xdotdot1 = empdata[:,1]

    wh_stroke = min(where(emptime>=tstroke)[0])
    wh_recovery = min(where(emptime>=trecovery)[0])

    xdotdot = 0*emptime
    xdotdot[:xdotdot.size-wh_recovery] = xdotdot1[wh_recovery:]
    xdotdot[xdotdot.size-wh_recovery:] = xdotdot1[:wh_recovery]

    wh_stroke = xdotdot.size-wh_recovery+wh_stroke

    xdot = cumsum(xdotdot)*empdt
    xdot = xdot-mean(xdot)+vavg

    Fdrag = drag_eq((Nrowers*mc)+mb,xdot,doprint=0,alfaref=alfa*dragform)

    ydotdot = 0*xdotdot
    zdotdot = 0*xdotdot


    # Recovery based
    ydotdot[0:wh_stroke] = (-Fdrag[0:wh_stroke]-(mb+(Nrowers*mc))*xdotdot[0:wh_stroke])/(Nrowers*mc)
    ydot = empdt*cumsum(ydotdot)

    
    Fhelp = mb*xdotdot+Fdrag

    # calculate phidot, phi
    phidot1 = xdot/lout
    phidot1[:wh_stroke] = 0
    phi1 = cumsum(phidot1)*empdt
    phi1 = phi1+catchangle - phi1[wh_stroke]

    phidot2 = xdot/(lout*np.cos(phi1))
    phidot2[:wh_stroke] = 0
    phi2 = cumsum(phidot2)*empdt
    phi2 = phi2+catchangle - phi2[wh_stroke]

    phidot3 = xdot/(lout*np.cos(phi2))
    phidot3[:wh_stroke] = 0
    phi3 = cumsum(phidot3)*empdt
    phi3 = phi3+catchangle - phi3[wh_stroke]

    phidot = xdot/(lout*np.cos(phi3))
    phidot[:wh_stroke] = 0
    phi = cumsum(phidot)*empdt
    phi = phi+catchangle - phi[wh_stroke]



    vhand = phidot*lin*np.cos(phi)
    vhand[:wh_stroke] = 0
    handlepos = cumsum(vhand)*empdt

    ydot[wh_stroke+1:] = crew.vcma(vhand[wh_stroke+1:],handlepos[wh_stroke+1:])

    ydotdot = gradient(ydot+xdot,empdtarray)

    zdot = (mc*(ydot+xdot)+mb*xdot)/(mc+mb)
    zdotdot = gradient(zdot,empdtarray)

    Fblade = (mc+mb)*zdotdot+Fdrag
    Fhandle = lout*Fblade/lin
    Ffoot = mc*(xdotdot+ydotdot)+Fhandle

    Pw = drag_eq((Nrowers*mc)+mb,xdot,alfaref=alfa*dragform)*xdot
    Edrag = cumsum(Pw)*empdt
    Pq = (Nrowers*mc)*(ydotdot)*ydot
    Pqrower = abs(Pq)
    Pdiss = Pqrower-Pq
    
    

    print(('Drag Power',mean(Pw)))
    print(('Kinetic Power loss',mean(Pdiss)))
    print(('Stroke length ',max(handlepos)))

    forcearray = transpose([handlepos[wh_stroke-1:],Fhandle[wh_stroke-1:]])

    savetxt('empforce.txt',forcearray,delimiter=',',fmt='%4.2e')


    recoveryarray = transpose([emptime[:wh_stroke-1],-ydot[:wh_stroke-1]])
    savetxt('emprecovery.txt',recoveryarray,delimiter=',',fmt='%4.2e')

    if (doplot==1):
      pyplot.clf()
      pyplot.plot(emptime,xdotdot, 'r-',label = 'Measured Boat Acceleration')
      pyplot.plot(emptime,ydotdot+xdotdot,'b-',label = 'Crew Acceleration')
      pyplot.plot(emptime,zdotdot,'g-',label = 'System Acceleration')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('a (m/s^2)')
      pyplot.show()

    if (doplot==2):
      pyplot.clf()
      pyplot.plot(emptime, xdot, 'r-',label = 'Boat Speed')
      pyplot.plot(emptime, xdot+ydot, 'b-',label = 'Crew Speed')
      pyplot.plot(emptime, zdot, 'g-',label = 'System speed')
      pylab.legend(loc='lower left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('v (m/s)')
      pyplot.show()

    if (doplot==3):
        pyplot.clf()
        pyplot.plot(emptime, Fdrag, 'r-',label = 'Drag Force')
        pyplot.plot(emptime, Fhelp, 'g-',label = 'Foar-Ffoot')
        pyplot.plot(emptime, Fblade, 'b-',label = 'Fblade')
        pyplot.plot(emptime, Ffoot,'y-', label = 'Ffoot')
        pyplot.plot(emptime, Fhandle,'k-', label = 'Fhandle')
        pylab.legend(loc='upper left')
        pyplot.xlabel("time (s)")
        pyplot.ylabel("F (N)")
        pyplot.show()
          
    if (doplot==4):
        pyplot.clf()
        pyplot.plot(emptime, phidot1, 'r-',label = 'Angular velocity')
        pyplot.plot(emptime, phidot2, 'g-',label = 'Iteration 2')
        pyplot.plot(emptime, phidot3, 'y-',label = 'Iteration 3')
        pyplot.plot(emptime, phidot, 'b-',label = 'Iteration 4')
        pylab.legend(loc='lower right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel("rad/s")
        pyplot.show()
          
    if (doplot==5):
        pyplot.clf()
        pyplot.plot(emptime, numpy.degrees(phi1), 'r-',label = 'Oar angle')
        pyplot.plot(emptime, numpy.degrees(phi2), 'g-',label = 'Iteration 2')
        pyplot.plot(emptime, numpy.degrees(phi3), 'y-',label = 'Iteration 3')
        pyplot.plot(emptime, numpy.degrees(phi), 'b-',label = 'Iteration 4')
        pylab.legend(loc='upper left')
        pyplot.xlabel("time (s)")
        pyplot.ylabel("degrees")
        pyplot.show()
          
       
    if (doplot==6):
        pyplot.clf()
        pyplot.plot(emptime, handlepos, 'r-', label = 'Handle position')
        pylab.legend(loc='upper left')
        pyplot.xlabel("time (s)")
        pyplot.ylabel("y (m)")

        pyplot.show()

    if (doplot==7):
        pyplot.clf()
        pyplot.plot(handlepos,Fhandle,'r-', label = 'Handle Force')
        pylab.legend(loc='upper left')
        pyplot.xlabel("x (m)")
        pyplot.ylabel("F (N)")
        pyplot.show()

    return mean(Pw+Pdiss)


def energybalance(F,crew,rigging,v0=4.3801,dt=0.03,doplot=1,doprint=0,
                  timewise=0,index_offset=1,empirical=0,empt0=0,vb0=0,
                  catchacceler=5.0,emptype='acceler',
                  windv=0,dowind=1):
    """ calculates one stroke with average handle force as input
    slide velocity and stroke/recovery ratio are calculated
    knows about slip, lift, drag. Plots energy balance.
    
    windv is wind speed in m/s. Positive values are tailwind.
    
    """

    # initialising output values
    dv = 100.
    vavg = 0.0
    vend = 0.0
    ratio = 0.0
    power = 0.0

    if (vb0==0):
        vb0 = v0

    if (catchacceler>50):
        catchacceler = 50

    # stroke parameters
    lin = rigging.lin
    lscull = rigging.lscull
    lout = lscull - lin
    tempo = crew.tempo
    mc = crew.mc
    mb = rigging.mb
    recprofile = crew.recprofile
    d = crew.strokelength
    Nrowers = rigging.Nrowers
    try:
        dragform = rigging.dragform
    except:
        dragform = 1.0

    catchacceler = max(catchacceler,2.0)

    # nr of time steps
    aantal = 1+int(round(60./(tempo*dt)))
    time = linspace(0,60./tempo,aantal)

    vs = zeros(len(time))+v0
    vb = zeros(len(time))+v0
    vc = zeros(len(time))+v0

    oarangle = zeros(len(time))
    xblade = zeros(len(time))
    Fhandle = zeros(len(time))
    Fblade = zeros(len(time))
    Fprop = zeros(len(time))
    Fhandle[0:2] = 0
   
    Pbladeslip = zeros(len(time))    # H

    xdotdot = zeros(len(time))
    zdotdot = zeros(len(time))
    ydotdot = zeros(len(time))

    xdot = zeros(len(time))+v0
    ydot = zeros(len(time))+v0
    zdot = zeros(len(time))+v0

    Pf = zeros(len(time))
    Foarlock = zeros(len(time))
    Flift = zeros(len(time))
    Fbldrag = zeros(len(time))
    attackangle = zeros(len(time))
    Clift = zeros(len(time))
    Cdrag = zeros(len(time))
   
    handlepos = 0

    # initial handle and boat velocities
    vs[0] = v0
    vb[0] = vb0
    vc[0] = ((Nrowers*mc+mb)*vs[0]-mb*vb[0])/(Nrowers*mc)
    oarangle[0] = rigging.oarangle(0)
    xblade[0] = -lout*np.sin(oarangle[0])


    i=1
   
    vcstroke = 0
    vcstroke2 = 1

    # catch
    vblade = xdot[i-1]

    while (vcstroke < vcstroke2):
        vhand = catchacceler*(time[i]-time[0])
        
        vcstroke = crew.vcm(vhand, handlepos)
        phidot = vb[i-1]*np.cos(oarangle[i-1])
        vhand = phidot*lin*np.cos(oarangle[i-1])
        ydot[i] = vcstroke
        Fdrag = drag_eq((Nrowers*mc)+mb,xdot[i-1],alfaref=alfa*dragform)
        zdotdot[i] = -Fdrag/((Nrowers*mc)+mb)
        vw = windv-vcstroke-zdot[i-1]
        Fwind = 0.5*crewarea*Cdw*rho_air*(Nrowers**scalepower)*vw*abs(vw)*dowind
        #       print(Fwind,crewarea,dowind)
        zdotdot[i] = zdotdot[i] + Fwind/((Nrowers*mc)+mb)
        zdot[i] = zdot[i-1]+dt*zdotdot[i]
        xdot[i] = zdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[i]

        Fi = crew.forceprofile(F,handlepos)
        Fbladei = Fi*lin/lout 
        res = blade_force(oarangle[i-1],rigging,vb[i-1],Fbladei)
        phidot2 = res[0]
        vhand2 = phidot2*lin*np.cos(oarangle[i-1])
        vcstroke2 = crew.vcm(vhand2,handlepos)

       
        vblade = xdot[i]-phidot*lout*np.cos(oarangle[i-1])
        #       print(i,vhand,vhand2,vcstroke,vcstroke2)
        vs[i] = zdot[i]
        vc[i] = xdot[i]+ydot[i]
        vb[i] = xdot[i]

        ydotdot[i] = (ydot[i]-ydot[i-1])/dt
        xdotdot[i] = zdotdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[i]


        handlepos = handlepos+ydot[i]*dt
        Fhandle[i] = 0
       
        oarangle[i] = rigging.oarangle(handlepos)
        i = i+1

    # stroke 
    while (handlepos<d) & (i<len(time)):
        if (timewise == 1):
            Fi = crew.forceprofile(F,handlepos)*np.cos(oarangle[i-1])
        else:
            Fi = crew.forceprofile(F,handlepos)
        Fhandle[i-1] = Fi
        Fblade[i-1] = Fi*lin/lout 
        res = blade_force(oarangle[i-1],rigging,vb[i-1],Fblade[i-1])
        phidot = res[0]

        Fprop[i-1] = res[2]*Nrowers
        Flift[i-1] = res[3]*Nrowers
        Fbldrag[i-1] = res[4]*Nrowers
        Clift[i-1] = res[5]
        Cdrag[i-1] = res[6]
        attackangle[i-1] = res[7]

        phidot = res[0]
        vhand = phidot*lin*np.cos(oarangle[i-1])

        vcstroke = crew.vcm(vhand, handlepos)
        Pbladeslip[i-1] = Nrowers*res[1]*(phidot*lout - vb[i-1]*np.cos(oarangle[i-1]))
        Fdrag = drag_eq((Nrowers*mc)+mb,xdot[i-1],alfaref=alfa*dragform)
        zdotdot[i] = (Fprop[i-1] - Fdrag)/((Nrowers*mc)+mb)

        vw = windv-vcstroke-zdot[i-1]
        Fwind = 0.5*crewarea*Cdw*rho_air*(Nrowers**scalepower)*vw*abs(vw)*dowind
        zdotdot[i] = zdotdot[i] + Fwind/((Nrowers*mc)+mb)

        zdot[i] = zdot[i-1]+dt*zdotdot[i]
      
          
        ydot[i] = vcstroke
        xdot[i] = zdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[i]

        handlepos = handlepos+vhand*dt
        vs[i] = zdot[i]
        vc[i] = xdot[i]+ydot[i]
        vb[i] = xdot[i]

        ydotdot[i] = (ydot[i]-ydot[i-1])/dt
        xdotdot[i] = zdotdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[i]

        Pf[i-1] = Nrowers*Fblade[i-1]*xdot[i]*np.cos(oarangle[i-1])

        oarangle[i] = rigging.oarangle(handlepos)  

        i = i+1
      
    i=i-1;

    # recovery

    trecovery = max(time)-time[i]

    ratio = time[i]/max(time)
    aantalstroke = i

    if (recprofile == 1): # oude methode (sinus)
        vhandmax = -np.pi*d/(2*trecovery)
        vhand = vhandmax*np.sin(np.pi*(time-time[i])/trecovery)
        for k in range(i+1,aantal):
            Fdrag = drag_eq((Nrowers*mc)+mb,xdot[k-1],alfaref=alfa*dragform)
            zdotdot[k] = (- Fdrag)/((Nrowers*mc)+mb)

            vw = windv-vcstroke-zdot[k-1]
            Fwind = 0.5*crewarea*Cdw*rho_air*(Nrowers**scalepower)*vw*abs(vw)*dowind
            zdotdot[k] = zdotdot[k] + Fwind/((Nrowers*mc)+mb)

            zdot[k] = zdot[k-1]+dt*zdotdot[k]
            ydot[k] = crew.vcm(vhand[k], handlepos)
            xdot[k] = zdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[k]

            vs[k] = zdot[k]
            vc[k] = xdot[k]+ydot[k]
            vb[k] = xdot[k]

            ydotdot[k] = (ydot[k]-ydot[k-1])/dt
            xdotdot[k] = zdotdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[k]
            handlepos = handlepos+vhand[k]*dt
            oarangle[k] = rigging.oarangle(handlepos)
      
    else:
        vavgrec = d/trecovery
        vcrecovery = zeros(aantal)
        for k in range(i+1,aantal):
            vhand = crew.vhandle(vavgrec,trecovery,time[k]-time[i])
            vcrecovery[k] = crew.vcm(vhand, handlepos)

            Fdrag = drag_eq((Nrowers*mc)+mb,xdot[k-1],alfaref=alfa*dragform)
            zdotdot[k] = (- Fdrag)/((Nrowers*mc)+mb)

            vw = windv-vcstroke-zdot[k-1]
            Fwind = 0.5*crewarea*Cdw*rho_air*(Nrowers**scalepower)*vw*abs(vw)*dowind
            zdotdot[k] = zdotdot[k] + Fwind/((Nrowers*mc)+mb)


            zdot[k] = zdot[k-1]+dt*zdotdot[k]
            ydot[k] = vcrecovery[k]
            xdot[k] = zdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[k]

            vs[k] = zdot[k]
            vc[k] = xdot[k]+ydot[k]
            vb[k] = xdot[k]

            ydotdot[k] = (ydot[k]-ydot[k-1])/dt
            xdotdot[k] = zdotdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[k]

            handlepos = d+d*crew.dxhandle(vavgrec,trecovery,time[k]-time[i])
            #         handlepos = handlepos+vhand*dt
            oarangle[k] = rigging.oarangle(handlepos)
      

    # blade positions      
    xblade=dt*cumsum(vb)-np.sin(oarangle)*lout
    yblade=lout*np.cos(oarangle)+rigging.spread


    # velocities
    xdot = vb
    zdot = vs
    ydot = vc-vb

    xdotdot[1]=(xdot[1]-xdot[0])/dt
    ydotdot[1]=(ydot[1]-ydot[0])/dt

    Pq = (Nrowers*mc)*(xdotdot+ydotdot)*ydot
   
    #   Ekinb = 0.5*mb*xdot**2 - 0.5*mb*v0**2
    #   Ekinc = 0.5*mc*(xdot+ydot)**2 - 0.5*mc*v0**2

    Pw = drag_eq((Nrowers*mc)+mb,xdot,alfaref=alfa*dragform)*xdot
    Pwmin = drag_eq((Nrowers*mc)+mb,mean(xdot),alfaref=alfa*dragform)*mean(xdot)

    Pmb = mb*xdot*xdotdot
    Pmc = (Nrowers*mc)*(xdot+ydot)*(xdotdot+ydotdot)

    #   Phandle = Nrowers*Fhandle*(xdot+ydot)*np.cos(oarangle)
    Phandle = Nrowers*Fhandle*(xdot)*np.cos(oarangle)

    Pleg = Nrowers*mc*(xdotdot+ydotdot)*ydot

    Ekinb = cumsum(Pmb)*dt
    Ekinc = cumsum(Pmc)*dt

    Pqrower = abs(Pq)
    Pdiss = Pqrower-Pq

    Ef = cumsum(Pf)*dt
    Eq = cumsum(Pq)*dt
    Eblade = cumsum(Pbladeslip)*dt
    Eqrower = cumsum(Pqrower)*dt
    Ediss = cumsum(Pdiss)*dt
    Ew = cumsum(Pw)*dt
    Ewmin = Pwmin*(max(time)-min(time))

    Eleg = cumsum(Pleg)*dt
    Ehandle = cumsum(Phandle)*dt
    Ekin0 = 0.5*(Nrowers*mc+mb)*zdot[0]**2
    Ekinend = 0.5*(Nrowers*mc+mb)*zdot[aantal-1]**2
    Eloss = Ekin0-Ekinend

    Fbltotal = (Fbldrag**2 + Flift**2)**(0.5)

    # empirical data
    
    if (empirical!=0):
        empdata = np.genfromtxt(empirical, delimiter = ',',skip_header=1)
        emptime = empdata[:,0]
        if (max(emptime)>10):
            emptime = emptime/1000.
        emptime = emptime + empt0
        empdt = emptime[1]-emptime[0]
        if (emptype == 'acceler'):
            empxdotdot = empdata[:,1]
            empxdot = cumsum(empxdotdot)*empdt
            empxdot = empxdot-mean(empxdot)+mean(xdot)
        else:
            empdtarray = gradient(emptime)
            empxdot = empdata[:,1]
            empxdotdot = gradient(empxdot,empdtarray)

        empRIM_E = max(cumsum(empxdot-min(empxdot))*empdt)
        empRIM_check = max(empxdot)-min(empxdot)
       
        if (doprint == 1):
            print(("RIM E (measured)",empRIM_E))
            print(("RIM Check (meas)",empRIM_check))
                     

    # some other calculations

    strokelength_cm = max(cumsum(ydot)*dt)


    # printing
    if (doprint==1):
        print(("E blade ",Eblade[aantal-1]))
        print(("Ediss rower ",Ediss[aantal-1]))
        print(("E drag ",Ew[aantal-1]))
        print(("Eleg ",Eleg[aantal-1]))
        print(("Ehandle ",Ehandle[aantal-1]))
        print(("Epropulsion ",Ef[aantal-1]))
        print(("Ekin loss ",Eloss))
        print("")
        print(("P blade ",Eblade[aantal-1]/time[aantal-1]))
        print(("P leg   ",Eleg[aantal-1]/time[aantal-1]))
        print(("P handle ",Ehandle[aantal-1]/time[aantal-1]))
        print(("P drag ",Ew[aantal-1]/time[aantal-1]))
        print(("P propulsion ",Ef[aantal-1]/time[aantal-1]))
        print("")
        print(("Stroke length CM ",strokelength_cm))
        print("")

    # plotting

    if (doplot==1):
        pyplot.clf()
        pyplot.subplot(111)

        pyplot.plot(time, xdot,'r-',label = 'Boat velocity')
        pyplot.plot(time, xdot+ydot,'g-',label = 'Crew velocity')
        pyplot.plot(time, zdot,'b-',label = 'CM velocity')
        if (empirical!=0):
            pyplot.plot(emptime, empxdot, 'y-',label = 'Measured')
        pylab.legend(loc='upper left')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('v (m/s)')

        pyplot.show()

    if(doplot==18):
        pyplot.clf()
        pyplot.plot(time,numpy.degrees(oarangle),'y.',label='oar angle')
        pylab.legend(loc='upper right')
        pyplot.ylabel("Oar Angle (o)")

        pyplot.show()


    if (doplot==2):
        pyplot.clf()
        pyplot.subplot(111)
        pyplot.plot(time, Pf,'r-',label = 'Propulsive power')
        pyplot.plot(time, Pq,'b-',label = 'Kinetic power')
        pyplot.plot(time, Pbladeslip,'k-',label = 'Puddle power')
        pyplot.plot(time, Pf+Pq+Pbladeslip,'g-',label = 'Leg power')
        pylab.legend(loc='upper right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('power (W)')
        pyplot.show()
      
    if (doplot==3):
        pyplot.clf()
        pyplot.subplot(111)
        pyplot.plot(time, Ef,'r-',label = 'Propulsive Energy')
        pyplot.plot(time, Eqrower,'b-',label = 'Kinetic Energy')
        pyplot.plot(time, Ef+Eqrower+Eblade,'g-',label = 'Total Energy')
        pyplot.plot(time, Eblade,'k-',label = 'Puddle Energy')
        pylab.legend(loc='upper left')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('energy (J)')
        pyplot.show()
      
    if (doplot==4):
        pyplot.clf()
        pyplot.subplot(111)
        pyplot.plot(time, Pw,'r-',label = 'Drag sink')
        pyplot.plot(time, Pbladeslip,'k-',label = 'Blade slip sink')
        pyplot.plot(time, Pmb,'b-',label = 'Kinetic energy change boat')
        pyplot.plot(time, Pmc,'g-',label = 'Kinetic energy change crew')
        pylab.legend(loc='upper right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('power (W)')
        pyplot.show()
      
    if (doplot==5):
        pyplot.clf()
        pyplot.subplot(111)
        pyplot.plot(time, Ew+Ediss+Eblade,'r-',label = 'Drag energy + Rower Diss + Blade Slip')
        pyplot.plot(time, Ew, 'y-', label = 'Drag Energy')
        pyplot.plot(time, Ekinb,'b-',label = 'Boat Kinetic energy')
        pyplot.plot(time, Ekinc,'g-',label = 'Crew Kinetic energy')
        pyplot.plot(time, Ew+Ediss+Ekinb+Ekinc+Eblade, 'k-', label = 'Ew + Ediss + Ekinb + Ekinc+Eblade')
        pylab.legend(loc='upper left')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('energy (J)')
        pyplot.show()
      
    if (doplot==6):
        pyplot.clf()
        pyplot.subplot(121)
        pyplot.plot(time, Pq,'k-',label = 'Kinetic power')
        pyplot.plot(time, 0*Pq, 'k-')
        pylab.legend(loc='upper right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('power (W)')

        pyplot.subplot(122)
        pyplot.plot(time, Pqrower,'b-',label = 'Kinetic power rower')
        pyplot.plot(time, Pdiss,'k-',label = 'Kinetic energy dissipation')
        pylab.legend(loc='upper right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('power (W)')

        pyplot.show()
      
    if (doplot==7):
        pyplot.clf()
        pyplot.plot(time, Ew+Ediss+Ekinb+Ekinc+Eblade, 'r-', label = 'Total Sinks')
        pyplot.plot(time, Ef+Eqrower+Eblade,'g-',label = 'Total Sources')
        pylab.legend(loc='lower right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('energy (J)')
        pyplot.show()

    if (doplot==8):
        pyplot.clf()
        pyplot.subplot(111)
        pyplot.plot(time, Phandle,'r-',label = 'Handle power (crew)')
        pyplot.plot(time, Pbladeslip,'g-',label = 'Puddle power')
        pyplot.plot(time, Pf, 'y-', label = 'Propulsive power')
        pyplot.plot(time, Pf+Pbladeslip,'k-',label = 'Propulsive+Puddle Power')
        pylab.legend(loc='upper right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('power (W)')
        pyplot.show()

    if (doplot==9):
        pyplot.clf()
        ax1 = pyplot.subplot(111)

        pyplot.plot(xblade,yblade,label='blade centre')
        
        pylab.legend(loc='best')
        pyplot.xlabel("x (m)")
        pyplot.ylabel('y (m)')
        ax1.axis('equal')

        xblade2 = xblade[0:len(xblade):4]
        yblade2 = yblade[0:len(xblade):4]
        oarangle2 = oarangle[0:len(xblade):4]

        for i in range(len(xblade2)):
            x1 = xblade2[i]+rigging.bladelength*np.sin(oarangle2[i])/2.
            x2 = xblade2[i]-rigging.bladelength*np.sin(oarangle2[i])/2.
            y1 = yblade2[i]-rigging.bladelength*np.cos(oarangle2[i])/2.
            y2 = yblade2[i]+rigging.bladelength*np.cos(oarangle2[i])/2.

        pyplot.plot([x1,x2],[y1,y2],'r-')

        pyplot.show()

    if (doplot==10):
        pyplot.clf()
        pyplot.plot(time, Fhandle, 'r-', label = 'Handle Force')
        pyplot.plot(time, Fblade, 'g-', label = 'Blade Force')
        pyplot.plot(time, Fprop, 'k-', label = 'Propulsive Force')
        pylab.legend(loc='lower right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel('Force (N)')
        pyplot.show()
        
    if (doplot==11):
        pyplot.clf()
        pyplot.plot(numpy.degrees(oarangle), Clift, 'r-', label = 'Lift coefficient')
        pyplot.plot(numpy.degrees(oarangle), Cdrag, 'g-', label = 'Drag coefficient')
        pylab.legend(loc='lower right')
        pyplot.xlabel("Oar Angle (degree)")
        pyplot.ylabel("Coefficient")
        pyplot.show()

    if (doplot==12):
        pyplot.clf()
        
        ax1 = pyplot.subplot(111)
        pyplot.plot(numpy.degrees(oarangle), Flift, 'r-', label = 'Lift Force')
        pyplot.plot(numpy.degrees(oarangle), Fbldrag, 'g-', label = 'Drag Force')
        pyplot.plot(numpy.degrees(oarangle), Fbltotal, 'k-', label = 'Total blade Force')
        pyplot.plot(numpy.degrees(oarangle),numpy.degrees(attackangle),'y.',label='angle of attack')
        pylab.legend(loc='lower right')
        pyplot.xlabel("Oar Angle (degree)")
        pyplot.ylabel("Blade Force")

        ax2 = pyplot.twinx()
        pyplot.plot(numpy.degrees(oarangle),numpy.degrees(attackangle),'y.',label='angle of attack')
        pylab.legend(loc='upper right')
        pyplot.ylabel("Angle of attack (o)")
        ax2.yaxis.tick_right()
        ax1 = pyplot.subplot(111)

        pyplot.show()


      
    if (doplot==13):
        pyplot.clf()
        pyplot.plot(time, ydot, 'r-', label = 'Crew velocity')
        pylab.legend(loc='lower right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel("v (m/s)")
        pyplot.show()
        
    if (doplot==14):
        pyplot.clf()
        pyplot.plot(time, xdotdot, 'r-', label = 'Boat acceleration')
        pyplot.plot(time, zdotdot, 'g-', label = 'System acceleration')
        pyplot.plot(time, ydotdot, 'b-', label = 'Crew acceleration')
        if (empirical!=0):
            pyplot.plot(emptime,empxdotdot, 'y-', label = 'Measured')
        pylab.legend(loc='upper right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel("Boat Acceleration (m/s2)")
        pyplot.show()

    if (doplot==15):
        pyplot.clf()
        pyplot.plot(time, ydot, 'r-', label = 'Recovery speed')
        pylab.legend(loc='upper right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel("Recovery Speed (m/s)")
        pyplot.show()

    if (doplot==16):
        pyplot.clf()
        pyplot.plot(time, numpy.degrees(oarangle), 'r-', label = 'Oar Angle')
        pylab.legend(loc='upper right')
        pyplot.xlabel("time (s)")
        pyplot.ylabel("Oar angle (o)")
        pyplot.show()

    if (doplot==19):
       pyplot.clf()
       ax1 = pyplot.subplot(111)
       pyplot.plot(time, xdotdot, 'r-', label = 'Boat acceleration')
       pyplot.plot(time, zdotdot, 'g-', label = 'System acceleration')
       pyplot.plot(time, ydotdot, 'b-', label = 'Crew acceleration')
       if (empirical!=0):
           pyplot.plot(emptime,empxdotdot, 'y-', label = 'Measured')
       pylab.legend(loc='upper right')
       pyplot.xlabel("time (s)")
       pyplot.ylabel("Boat Acceleration (m/s2)")

       ax2 = pyplot.twinx()
       pyplot.plot(time,numpy.degrees(oarangle),'y-',label='oar angle')
       pylab.legend(loc='upper left')
       pyplot.ylabel("Oar Angle (o)")
       ax2.yaxis.tick_right()

       pyplot.show()

    try:
        instanteff = (Pf+Pq)/(Pf+Pq+Pbladeslip)
    except RuntimeWarning:
        instanteff = 0.0
   
         

    if (doplot==17):
       pyplot.clf()
       pyplot.plot(time, instanteff, 'r-', label = 'Efficiency')
       pylab.legend(loc='upper right')
       pyplot.xlabel("time (s)")
       pyplot.ylabel("Efficiency")
       pyplot.show()

    # calculate check
    decel = -(abs(xdotdot[index_offset:])-xdotdot[index_offset:])/2.
    indices = decel.nonzero()
    decelmean = mean(decel[indices])
    cn_check = np.std(decel[indices])**2
       

    # calculate vavg, vmin, vmax, energy, efficiency, power
    dv = zdot[len(time)-1]-zdot[0]
    vavg = mean(xdot)
    vend = zdot[len(time)-1]
    energy = max(Ew+Ediss+Eblade-Eloss)
    efficiency = max(Ew-Eloss)/energy
    energy = energy/Nrowers
    power = energy*tempo/60.
    vmax = max(xdot)
    vmin = min(xdot)


    # calculate RIM parameters
    RIM_check = vmax-vmin
    RIM_E = max(cumsum(xdot-vmin)*dt)
    drag_eff = Ewmin/max(Ew)
    try:
        t4 = time[index_offset+min(where(decel==0)[0])]
        t3 = time[index_offset+max(where(decel==0)[0])]
    except ValueError:
        t4 = 1.0
        t3 = t4
    amin = min(xdotdot[2:])
    RIM_catchE = -(amin/t4)
    RIM_catchD = t4+max(time)-t3
    
    catchacceler = max(5,ydotdot[aantal-1]-xdotdot[aantal-1])

   
    return [dv,vend,vavg,ratio,energy,power,efficiency,vmax,vmin,cn_check,RIM_E,RIM_check,RIM_catchE,RIM_catchD,catchacceler,drag_eff]

def energybalance_erg(ratio,crew,erg,w0=4.3801,dt=0.03,doplot=1,doprint=0,theconst=1.0):
   """
   calculates one stroke with ratio as input, using force profile in time domain

   """

   # w0 = initial flywheel angular velo


   # initialising output values
   dv = 100.
   vavg = 0.0
   vend = 0.0
   power = 0.0

  # stroke parameters
   tempo = crew.tempo
   mc = crew.mc
   recprofile = crew.recprofile
   d = crew.strokelength
   Nrowers = 1
   drag = erg.drag
   inertia = erg.inertia
   cord = erg.cord
   cordlength = erg.cordlength
   r = erg.r # sprocket radius

   # nr of time steps
   aantal = 1+int(round(60./(tempo*dt)))
   time = linspace(0,60./tempo,aantal)

   # flywheel angular velo
   wf = zeros(len(time))+w0
   wfdot = zeros(len(time))
   # crew velo
   vc = zeros(len(time))
   vpull = zeros(len(time))

   Fhandle = zeros(len(time))
   Fres = zeros(len(time))
   Fleg = zeros(len(time))

   ydotdot = zeros(len(time))

   ydot = zeros(len(time)) # +wf[0]*r

   Pf = zeros(len(time))
   Phandle = zeros(len(time))

   Ebungee = zeros(len(time))
   Pbungee = zeros(len(time))
   
   handlepos = 0
   vhand = ydot[0]

   # initial handle and boat velocities
   vc[0] = ydot[0]

   # calculate average drive speed
   tdrive = ratio*max(time)
   vdriveavg = crew.strokelength/tdrive

   idrivemax = int(round(tdrive/dt))

##   powerconst = 2.58153699   # bij sin^(1/3)
##   powerconst = 2   # bij sin
#   powerconst = 1.5708 # bij sin^2
#   macht = 2.

#   vhandmax = np.pi*d/(powerconst*tdrive)
#   vhand = vhandmax*(np.sin(np.pi*(time)/tdrive))**(macht)

#   powerconst = 3.1733259127
#   vhandmax = np.pi*d/(powerconst*tdrive)
#   vhand = vhandmax*(1-np.cos(2*np.pi*(time)/tdrive))

   macht = 0.5
   x = np.linspace(0,1,100)
   y = (x-x**2)**(macht)

   s = np.cumsum(np.diff(x)*y[1:])[-1]

   powerconst = 1/s

   vhandmax = powerconst*d/tdrive
   vhand = vhandmax*((time/tdrive)-(time/tdrive)**2)**macht

   # stroke
   for i in range(1,idrivemax):

       now = dt*i
       timerel = now/tdrive
       time2 = (dt*(i+1))/tdrive
       vi = vhand[i-1]
       vj = vhand[i]
       vpull[i] = vhand[i]
       Tdrag = drag*wf[i-1]**2

       handlepos += dt*vi
      
       ydot[i] = crew.vcm(vi, handlepos)
#       ydot[i] = vi*(1-timerel)
#       ydot[i] = vi
       ydotdot[i] = (ydot[i]-ydot[i-1])/dt


       wnext = vj/r
       wnext2 = wf[i-1]-dt*Tdrag/inertia
#       if wnext > 0.99*wf[i-1]:
       if wnext > wnext2:
           wf[i] = wnext
           Tacceler = inertia*(wnext-wf[i-1])/dt
       else:
           wf[i] = wf[i-1]-dt*Tdrag/inertia
           Tacceler = 0
           Tdrag = 0

       wfdot[i] = (wf[i]-wf[i-1])/dt
       
       Fhandle[i] = ((Tdrag+Tacceler)/r)+cord*(cordlength+handlepos)
       Fres[i] = Nrowers*mc*ydotdot[i]
       Fleg[i] = Fres[i]+Fhandle[i]

       Ebungee[i] = 0.5*(cord*(cordlength+handlepos)**2 - cord*cordlength**2)
       Pbungee[i] = (Ebungee[i]-Ebungee[i-1])/dt
       vc[i] = ydot[i]



   # recovery


   trecovery = max(time)-time[idrivemax]

   ratio = time[idrivemax]/max(time)
   aantalstroke = idrivemax

   if (recprofile == 1): # oude methode (sinus)
      vhandmax = -np.pi*d/(2*trecovery)
      vhand = vhandmax*np.sin(np.pi*(time-time[i])/trecovery)
      for k in range(idrivemax,aantal):
         Tdrag = drag*wf[k-1]**2  # drag torque
         wf[k] = wf[k-1]-dt*Tdrag/inertia
         ydot[k] = crew.vcm(vhand, handlepos)
#         ydot[k] = vhand
         vc[k] = ydot[k]
         ydotdot[k] = (ydot[k]-ydot[k-1])/dt
         handlepos = handlepos+vhand[k]*dt
         Ebungee[k] = 0.5*(cord*(cordlength+handlepos)**2 - cord*cordlength**2)
         Pbungee[k] = (Ebungee[k]-Ebungee[k-1])/dt

      
   else:
      vavgrec = d/trecovery
      vcrecovery = zeros(aantal)
      for k in range(idrivemax,aantal):
         vhand = crew.vhandle(vavgrec,trecovery,time[k]-time[idrivemax])
         vpull[k] = vhand
         vcrecovery[k] = crew.vcm(vhand, handlepos)
#        vcrecovery[k] = vhand
         Tdrag = drag*wf[k-1]**2  # drag torque
         wf[k] = wf[k-1]-dt*Tdrag/inertia
         wfdot[k] = (wf[k]-wf[k-1])/dt


         ydot[k] = vcrecovery[k]
         vc[k] = ydot[k]

         ydotdot[k] = (ydot[k]-ydot[k-1])/dt

         handlepos = d+d*crew.dxhandle(vavgrec,trecovery,time[k]-time[idrivemax])

                
         Fhandle[k] = cord*(cordlength+handlepos)
         Fres[k] = Nrowers*mc*ydotdot[k]
         Fleg[k] = Fres[k]+Fhandle[k]

         Ebungee[k] = 0.5*(cord*(cordlength+handlepos)**2 - cord*cordlength**2)
         Pbungee[k] = (Ebungee[k]-Ebungee[k-1])/dt

      
   ydot[0] = ydot[0]/2.
   ydotdot[1]=(ydot[1]-ydot[0])/dt

   Pq = (Nrowers*mc)*ydotdot*ydot


   Pleg = Fleg*ydot
   Phandle = Fhandle*vpull
   Parm = Phandle-Fhandle*ydot

   Plegdiss = 0.5*theconst*(abs(Pleg)-Pleg)
   Plegsource = abs(Pleg)

   Parmdiss = 0.5*theconst*(abs(Parm)-Parm)
   Parmsource = abs(Parm)

# sources

   Elegsource = cumsum(Plegsource)*dt
   Earmsource = cumsum(Parmsource)*dt

   Eleg = cumsum(Pleg)*dt
   Earm = cumsum(Parm)*dt

   Ehandle = cumsum(Phandle)*dt

# sinks   
# drag power 
   Pw = drag*wf**3.
   Ew = cumsum(Pw)*dt

   Elegdiss = cumsum(Plegdiss)*dt
   Earmdiss = cumsum(Parmdiss)*dt

# storage

   Pwheel = inertia*wf*wfdot
   Ewheel = cumsum(Pwheel)*dt
   Ewheel = Ewheel - Ewheel[0]

   Ebungee = cumsum(Pbungee)*dt

   Pqrower = abs(Pq)
   Pdiss = 0.5*theconst*(Pqrower-Pq)

   Eq = cumsum(Pq)*dt
   Eqrower = cumsum(Pqrower)*dt
   Ediss = cumsum(Pdiss)*dt


   # printing
   if (doprint==1):
      print(("Ediss rower ",Ediss[aantal-1]))
      print(("E drag ",Ew[aantal-1]))
      print(("Eleg ",Eqrower[aantal-1]))
      print(("Ehandle ",Ehandle[aantal-1]))
      print(("Ebungee ",Ebungee[aantal-1]))
      print("")
      print(("P handle ",Ehandle[aantal-1]/time[aantal-1]))
      print(("P drag ",Ew[aantal-1]/time[aantal-1]))
      print("")

   # plotting

   if (doplot==1):
      pyplot.clf()
      pyplot.subplot(111)

      pyplot.plot(time, ydot,'r-',label = 'Crew velocity')
      pyplot.plot(time, vpull,'k-',label = 'Handle velocity')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('v (m/s)')


      pyplot.show()

   if (doplot==2):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Fhandle,'r-',label = 'Handle force')
      pyplot.plot(time, Fleg,'b-',label = 'Leg force')
      pyplot.plot(time, Fres,'g-',label = 'Accelerating force')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('force (N)')
      pyplot.show()
      
   if (doplot==3):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Phandle, 'r-', label = 'Handle Power')
      pyplot.plot(time, Pleg,'b-',label = 'Leg power')
      pyplot.plot(time, Pq,'k-',label = 'Kinetic power')
      pyplot.plot(time, Parm,'y-',label = 'Arm power')
      pyplot.plot(time, Pq+Phandle-Parm-Pleg,'b+', label = 'should be zero')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()
      
   if (doplot==4):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Ewheel,'g-',label = 'Flywheel energy stored')
      pyplot.plot(time, Eq+Ebungee,'k-',label = 'Kinetic energy')
      pyplot.plot(time, Ew,'r-',label = 'Drag dissipation')
      pyplot.plot(time, Ediss,'b-',label = 'Rower body dissipation')
      pyplot.plot(time, Ewheel+Eq+Ew+Ediss+Ebungee, 'b+', label = 'Sinks+Kinetic')
      pyplot.plot(time, Ew+Ediss, 'r+', label = 'Sinks')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('Energy (J)')
      pyplot.show()
      
   if (doplot==5):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Pleg, 'y-', label = 'Leg power')
      pyplot.plot(time, Plegdiss,'g-',label = 'Leg dissipation')
      pyplot.plot(time, Plegsource,'g+',label = 'Leg source')
      pyplot.plot(time, Parm, 'r-', label = 'Arm power')
      pyplot.plot(time, Parmdiss,'k-',label = 'Arm dissipation')
      pyplot.plot(time, Parmsource,'k+',label = 'Arm source')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()

   if (doplot==6):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Elegsource+Ehandle, 'bo', label = 'Leg power')
      pyplot.plot(time, Elegdiss,'g-',label = 'Leg dissipation')
      pyplot.plot(time, Earm, 'r-', label = 'Arm power')
      pyplot.plot(time, Ehandle, 'k+', label = 'Handle power')
      pyplot.plot(time, Earmdiss,'k-',label = 'Arm dissipation')
      pyplot.plot(time, Eqrower+Ewheel+Ebungee, 'y+', label = 'Eqrower+Ewheel+Ecord')
      pyplot.plot(time, Elegsource+Earmsource,'b+', label = 'Sources')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()
      
      
      
   if (doplot==7):
      pyplot.clf()
      pyplot.plot(time, Ew+Ediss, 'r-', label = 'Total Sinks')
#      pyplot.plot(time, Elegsource+Earmsource,'go',label = 'Total Sources')
      pyplot.plot(time, Eqrower+Ehandle,'y-',label = 'Total Sources 2')
      pyplot.plot(time, Ewheel+Eq+Ew+Ediss+Ebungee, 'b+', label = 'Sinks+Kinetic')
      pylab.legend(loc='lower right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()


   if (doplot==8):
      pyplot.clf()
      pyplot.plot(time, ydot, 'r-', label = 'Crew velocity')
      pylab.legend(loc='lower right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel("v (m/s)")
      pyplot.show()

   if (doplot==9):
      pyplot.clf()
      wref = wf
      pyplot.plot(time,wref,'r-',label='flywheel speed')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel("Flywheel speed (rad/sec)")
      pyplot.show()


   dw = wf[len(time)-1]-wf[0]
   wavg = mean(wf)
   wend = wf[len(time)-1]
   energy = max(Ew+Ediss)
   energyd = max(Ew)
   energy = energy/Nrowers
   energyd = energyd/Nrowers
   power = energy*tempo/60.
   powerd = energyd*tempo/60.
   
   return [dw,wend,wavg,ratio,energy,power,powerd]



def energybalance_erg_old(F,crew,erg,w0=4.3801,dt=0.03,doplot=1,doprint=0,
                  timewise=0,theconst=1.0):
   # calculates one stroke with average handle force as input

   # w0 = initial flywheel angular velo


   # initialising output values
   dv = 100.
   vavg = 0.0
   vend = 0.0
   ratio = 0.0
   power = 0.0

  # stroke parameters
   tempo = crew.tempo
   mc = crew.mc
   recprofile = crew.recprofile
   d = crew.strokelength
   Nrowers = 1
   drag = erg.drag
   inertia = erg.inertia
   cord = erg.cord
   cordlength = erg.cordlength
   r = erg.r # sprocket radius

   # nr of time steps
   aantal = 1+int(round(60./(tempo*dt)))
   time = linspace(0,60./tempo,aantal)

   # flywheel angular velo
   wf = zeros(len(time))+w0
   # crew velo
   vc = zeros(len(time))

   Fhandle = zeros(len(time))

   ydotdot = zeros(len(time))

   ydot = zeros(len(time)) # +wf[0]*r

   Pf = zeros(len(time))

   Ebungee = zeros(len(time))
   Pbungee = zeros(len(time))
   
   handlepos = 0
   vhand = ydot[0]

   # initial handle and boat velocities
   vc[0] = ydot[0]

   i=1

   # stroke
   while handlepos<d:
       if (timewise == 1):
           Fi = crew.forceprofile(F,handlepos)
       else:
           Fi = crew.forceprofile(F,handlepos)
      
       Fhandle[i-1] = Fi
       Tdrag = drag*wf[i-1]**2  # drag torque

       if (0.999*wf[i-1]*r > vhand):
           if (ydot[i-1] != 0):
               maxforce = crew.maxpower/ydot[i-1]
           else:
               maxforce = crew.maxforce
           if maxforce<Fhandle[i-1]:
               maxforce=Fhandle[i-1]
           thepower = ydot[i-1]*maxforce
           if (thepower==0):
               thepower = crew.maxpower
           kinerg = 0.5*Nrowers*mc*(ydot[i-1])**2
           kinerg = kinerg+dt*thepower
           ydot[i] = np.sqrt(2*kinerg/(Nrowers*mc))
           ydotdot[i] = (ydot[i]-ydot[i-1])/dt
           if ydot[i]>wf[i-1]*r:
               ydot[i]=wf[i-1]*r
           dw = dt*(-Tdrag)/inertia
           wf[i] = wf[i-1]+dw
           vhand = crew.vha(ydot[i], handlepos)
       else:
           dw = dt*(Fhandle[i-1]*r-Tdrag)/inertia
           wf[i] = wf[i-1]+dw
           vhand = wf[i]*r
           vcstroke = crew.vcm(vhand, handlepos)
           ydot[i] = vcstroke

       handlepos = handlepos+vhand*dt
       Ebungee[i] = 0.5*(cord*(cordlength+handlepos)**2 - cord*cordlength**2)
       Pbungee[i] = (Ebungee[i]-Ebungee[i-1])/dt
       vc[i] = ydot[i]

       ydotdot[i] = (ydot[i]-ydot[i-1])/dt


       Pf[i-1] = Nrowers*Fhandle[i-1]*ydot[i]

       i = i+1
      
   i=i-1;

   # recovery


   trecovery = max(time)-time[i]

   ratio = time[i]/max(time)
   aantalstroke = i

   if (recprofile == 1): # oude methode (sinus)
      vhandmax = -np.pi*d/(2*trecovery)
      vhand = vhandmax*np.sin(np.pi*(time-time[i])/trecovery)
      for k in range(i+1,aantal):
         Tdrag = drag*wf[k-1]**2  # drag torque
         wf[k] = wf[k-1]-dt*Tdrag/inertia
         ydot[k] = crew.vcm(vhand, handlepos)
         vc[k] = ydot[k]

         ydotdot[k] = (ydot[k]-ydot[k-1])/dt
         handlepos = handlepos+vhand[k]*dt
         Ebungee[k] = 0.5*(cord*(cordlength+handlepos)**2 - cord*cordlength**2)
         Pbungee[k] = (Ebungee[k]-Ebungee[k-1])/dt

      
   else:
      vavgrec = d/trecovery
      vcrecovery = zeros(aantal)
      for k in range(i+1,aantal):
         vhand = crew.vhandle(vavgrec,trecovery,time[k]-time[i])
         vcrecovery[k] = crew.vcm(vhand, handlepos)
         Tdrag = drag*wf[k-1]**2  # drag torque

         wf[k] = wf[k-1]-dt*Tdrag/inertia
         ydot[k] = vcrecovery[k]
         vc[k] = ydot[k]

         ydotdot[k] = (ydot[k]-ydot[k-1])/dt

         handlepos = d+d*crew.dxhandle(vavgrec,trecovery,time[k]-time[i])
         Ebungee[k] = 0.5*(cord*(cordlength+handlepos)**2 - cord*cordlength**2)
         Pbungee[k] = (Ebungee[k]-Ebungee[k-1])/dt

      
   ydot[0] = ydot[0]/2.
   ydotdot[1]=(ydot[1]-ydot[0])/dt

   Pq = (Nrowers*mc)*ydotdot*ydot
   Pq = Pq+Pbungee
   
# drag power
   Pw = drag*wf**3.

   Phandle = Nrowers*Fhandle*wf*r

   Pqrower = abs(Pq)
   Pdiss = 0.5*theconst*(Pqrower-Pq)

   Eq = cumsum(Pq)*dt
   Eqrower = cumsum(Pqrower)*dt

   Ew = cumsum(Pw)*dt
   Ehandle = cumsum(Phandle)*dt
   Ediss = cumsum(Pdiss)*dt

   Ekinrower = 0.5*Nrowers*mc*ydot**2
   Ekinerg = 0.5*inertia*wf**2

   Ekinrower = Ekinrower
   Ekinerg = Ekinerg-0.5*inertia*w0**2

   # printing
   if (doprint==1):
      print(("Ediss rower ",Ediss[aantal-1]))
      print(("E drag ",Ew[aantal-1]))
      print(("Eleg ",Eqrower[aantal-1]))
      print(("Ehandle ",Ehandle[aantal-1]))
      print(("Ebungee ",Ebungee[aantal-1]))
      print("")
      print(("P handle ",Ehandle[aantal-1]/time[aantal-1]))
      print(("P drag ",Ew[aantal-1]/time[aantal-1]))
      print("")

   # plotting

   if (doplot==1):
      pyplot.clf()
      pyplot.subplot(111)

      pyplot.plot(time, ydot,'r-',label = 'Crew velocity')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('v (m/s)')


      pyplot.show()

   if (doplot==2):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Phandle,'r-',label = 'Handle power')
      pyplot.plot(time, Pq,'b-',label = 'Kinetic power')
      pyplot.plot(time, Phandle+Pq,'g-',label = 'Leg power')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()
      
   if (doplot==3):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Ehandle,'r-',label = 'Propulsive Energy')
      pyplot.plot(time, Eqrower,'b-',label = 'Kinetic Energy')
      pyplot.plot(time, Ebungee,'k-',label = 'Bungee cord energy')
      pyplot.plot(time, Ehandle+Eqrower,'g-',label = 'Total Energy')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()
      
   if (doplot==4):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Eq,'g-',label = 'Kinetic energy change crew')
      pyplot.plot(time, Ediss,'r-',label = 'Kinetic energy change crew (lost)')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('Energy (J)')
      pyplot.show()
      
   if (doplot==5):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Ew, 'y-', label = 'Drag Energy')
      pyplot.plot(time, Ediss,'g-',label = 'Crew Kinetic energy lost')
      pyplot.plot(time, Ew+Ediss, 'k-', label = 'Ew + Ediss')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()
      
   if (doplot==6):
      pyplot.clf()
      pyplot.subplot(121)
      pyplot.plot(time, Pq,'k-',label = 'Kinetic power')
      pyplot.plot(time, 0*Pq, 'k-')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')

      pyplot.subplot(122)
      pyplot.plot(time, Pqrower,'b-',label = 'Kinetic power rower')
      pyplot.plot(time, Pdiss,'k-',label = 'Kinetic energy dissipation')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')

      pyplot.show()
      
   if (doplot==7):
      pyplot.clf()
      pyplot.plot(time, Ew+Ediss, 'r-', label = 'Total Sinks')
      pyplot.plot(time, Ehandle+Eqrower,'g-',label = 'Total Sources')
      pyplot.plot(time, Ew+Ediss+Ekinrower+Ekinerg+Ebungee,'k-',label = 'Total Sinks+Stored')
      pylab.legend(loc='lower right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()


   if (doplot==8):
      pyplot.clf()
      pyplot.plot(time, ydot, 'r-', label = 'Crew velocity')
      pylab.legend(loc='lower right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel("v (m/s)")
      pyplot.show()

   if (doplot==9):
      pyplot.clf()
      wref = wf/(2.0*np.pi)
      pyplot.plot(time,wref,'r-',label='flywheel speed')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel("Flywheel speed (rev/sec)")
      pyplot.show()


   dw = wf[len(time)-1]-wf[0]
   wavg = mean(wf)
   wend = wf[len(time)-1]
   energy = max(Ew+Ediss)
   energyd = max(Ew)
   energy = energy/Nrowers
   energyd = energyd/Nrowers
   power = energy*tempo/60.
   powerd = energyd*tempo/60.
   
   return [dw,wend,wavg,ratio,energy,power,powerd]



def energybalance_old(F,crew,rigging,v0=4.3801,dt=0.03,doplot=1,doprint=0,
                  timewise=0):
   # calculates one stroke with average handle force as input
   # slide velocity and stroke/recovery ratio are calculated
   # knows about slip, lift, drag. Plots energy balance.


   # initialising output values
   dv = 100.
   vavg = 0.0
   vend = 0.0
   ratio = 0.0
   power = 0.0

  # stroke parameters
   lin = rigging.lin
   lscull = rigging.lscull
   lout = lscull - lin
   tempo = crew.tempo
   mc = crew.mc
   mb = rigging.mb
   recprofile = crew.recprofile
   d = crew.strokelength
   Nrowers = rigging.Nrowers
   try:
       dragform = rigging.dragform
   except:
       dragform = 1.0

   alfaboat = alfa*((Nrowers*mc+mb)/(94.0))**(2./3.)


   # nr of time steps
   aantal = 1+int(round(60./(tempo*dt)))
   time = linspace(0,60./tempo,aantal)

   vs = zeros(len(time))+v0
   vb = zeros(len(time))+v0
   vc = zeros(len(time))+v0

   oarangle = zeros(len(time))
   xblade = zeros(len(time))
   Fhandle = zeros(len(time))
   Fblade = zeros(len(time))
   Fprop = zeros(len(time))
   Fhandle[0:2] = F
   Pbladeslip = zeros(len(time))    # H

   xdotdot = zeros(len(time))
   zdotdot = zeros(len(time))
   ydotdot = zeros(len(time))

   xdot = zeros(len(time))+v0
   ydot = zeros(len(time))+v0
   zdot = zeros(len(time))+v0

   Pf = zeros(len(time))
   Foarlock = zeros(len(time))
   
   handlepos = 0

   # initial handle and boat velocities
   vs[0] = v0
   vb[0] = vs[0]+0.0
   vc[0] = vs[0]+0.0
   oarangle[0] = rigging.oarangle(0)
   xblade[0] = -lout*np.sin(oarangle[0])


   i=1

   # stroke
   while handlepos<d:
      if (timewise == 1):
         Fi = crew.forceprofile(F,handlepos)*np.cos(oarangle[i-1])
      else:
         Fi = crew.forceprofile(F,handlepos)
      Fhandle[i-1] = Fi
      Fblade[i-1] = Fi*lin/lout 
      res = blade_force(oarangle[i-1],rigging,vb[i-1],Fblade[i-1])
      phidot = res[0]
#      for u in range(5):
##         l2 = lout-phidot/(vb[i-1]*np.cos(oarangle[i-1]))
#         l2 = lout
#         Fblade[i-1] = Fhandle[i-1]*lin/l2
#         res = blade_force(oarangle[i-1],rigging,vb[i-1],Fblade[i-1])
#         phidot = res[0]

      Fprop[i-1] = res[2]*Nrowers
      phidot = res[0]
      vhand = phidot*lin*np.cos(oarangle[i-1])
#      vcstroke = vhand*(1-(handlepos/d))
      vcstroke = crew.vcm(vhand, handlepos)
      Pbladeslip[i-1] = Nrowers*res[1]*(phidot*lout - vb[i-1]*np.cos(oarangle[i-1]))

      Fdrag = alfaboat*xdot[i-1]**2
#      Fdrag = drag_eq((Nrowers*mc)+mb,xdot[i-1])
      zdotdot[i] = (Fprop[i-1] - Fdrag)/((Nrowers*mc)+mb)
      zdot[i] = zdot[i-1]+dt*zdotdot[i]
      ydot[i] = vcstroke
      xdot[i] = zdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[i]

      handlepos = handlepos+vhand*dt
      vs[i] = zdot[i]
      vc[i] = xdot[i]+ydot[i]
      vb[i] = xdot[i]

      ydotdot[i] = (ydot[i]-ydot[i-1])/dt
      xdotdot[i] = zdotdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[i]

      Pf[i-1] = Nrowers*Fblade[i-1]*xdot[i]*np.cos(oarangle[i-1])

      oarangle[i] = rigging.oarangle(handlepos)  

      i = i+1
      
   i=i-1;

   # recovery

   trecovery = max(time)-time[i]

   ratio = time[i]/max(time)
   aantalstroke = i

   if (recprofile == 1): # oude methode (sinus)
      vhandmax = -np.pi*d/(2*trecovery)
      vhand = vhandmax*np.sin(np.pi*(time-time[i])/trecovery)
      for k in range(i+1,aantal):
         Fdrag = alfaboat*xdot[k-1]**2
#         Fdrag = drag_eq((Nrowers*mc)+mb,xdot[k-1])
         zdotdot[k] = (- Fdrag)/((Nrowers*mc)+mb)
         zdot[k] = zdot[k-1]+dt*zdotdot[k]
         ydot[k] = crew.vcm(vhand[k], handlepos)
         xdot[k] = zdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[k]

         vs[k] = zdot[k]
         vc[k] = xdot[k]+ydot[k]
         vb[k] = xdot[k]

         ydotdot[k] = (ydot[k]-ydot[k-1])/dt
         xdotdot[k] = zdotdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[k]
         handlepos = handlepos+vhand[k]*dt
         oarangle[k] = rigging.oarangle(handlepos)
      
   else:
      vavgrec = d/trecovery
      vcrecovery = zeros(aantal)
      for k in range(i+1,aantal):
         vhand = crew.vhandle(vavgrec,trecovery,time[k]-time[i])
         vcrecovery[k] = crew.vcm(vhand, handlepos)

         Fdrag = alfaboat*xdot[k-1]**2
#         Fdrag = drag_eq((Nrowers*mc)+mb,xdot[k-1])
         zdotdot[k] = (- Fdrag)/((Nrowers*mc)+mb)
         zdot[k] = zdot[k-1]+dt*zdotdot[k]
         ydot[k] = vcrecovery[k]
         xdot[k] = zdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[k]

         vs[k] = zdot[k]
         vc[k] = xdot[k]+ydot[k]
         vb[k] = xdot[k]

         ydotdot[k] = (ydot[k]-ydot[k-1])/dt
         xdotdot[k] = zdotdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[k]

         handlepos = d+d*crew.dxhandle(vavgrec,trecovery,time[k]-time[i])
#         handlepos = handlepos+vhand*dt
         oarangle[k] = rigging.oarangle(handlepos)
      

# blade positions      
   xblade=dt*cumsum(vb)-np.sin(oarangle)*lout
   yblade=lout*np.cos(oarangle)+rigging.spread


# velocities
   xdot = vb
   zdot = vs
   ydot = vc-vb

   xdotdot[1]=(xdot[1]-xdot[0])/dt
   ydotdot[1]=(ydot[1]-ydot[0])/dt

   Pq = (Nrowers*mc)*(xdotdot+ydotdot)*ydot
   
   Ekinb1 = 0.5*mb*xdot**2 - 0.5*mb*v0**2
   Ekinc1 = 0.5*mc*(xdot+ydot)**2 - 0.5*mc*v0**2

#   Pw = drag_eq((Nrowers*mc)+mb,xdot)*xdot
   Pw = alfaboat*xdot**3
   print((alfaboat,xdot[0],Pw[0]))

   Pmb = mb*xdot*xdotdot
   Pmc = (Nrowers*mc)*(xdot+ydot)*(xdotdot+ydotdot)

   Phandle = Nrowers*Fhandle*(xdot+ydot)*np.cos(oarangle)

   Pleg = Nrowers*mc*(xdotdot+ydotdot)*ydot

   Ekinb = cumsum(Pmb)*dt
   Ekinc = cumsum(Pmc)*dt

   Pqrower = abs(Pq)
   Pdiss = Pqrower-Pq

   Ef = cumsum(Pf)*dt
   Eq = cumsum(Pq)*dt
   Eblade = cumsum(Pbladeslip)*dt
   Eqrower = cumsum(Pqrower)*dt
   Ediss = cumsum(Pdiss)*dt
   Ew = cumsum(Pw)*dt
   Eleg = cumsum(Pleg)*dt
   Ehandle = cumsum(Phandle)*dt
   Ekin0 = 0.5*(Nrowers*mc+mb)*zdot[0]**2
   Ekinend = 0.5*(Nrowers*mc+mb)*zdot[aantal-1]**2
   Eloss = Ekin0-Ekinend

   # printing
   if (doprint==1):
      print(("E blade ",Eblade[aantal-1]))
      print(("Ediss rower ",Ediss[aantal-1]))
      print(("E drag ",Ew[aantal-1]))
      print(("Eleg ",Eleg[aantal-1]))
      print(("Ehandle ",Ehandle[aantal-1]))
      print(("Epropulsion ",Ef[aantal-1]))
      print(("Ekin loss ",Eloss))
      print("")
      print(("P blade ",Eblade[aantal-1]/time[aantal-1]))
      print(("P leg   ",Eleg[aantal-1]/time[aantal-1]))
      print(("P handle ",Ehandle[aantal-1]/time[aantal-1]))
      print(("P drag ",Ew[aantal-1]/time[aantal-1]))
      print(("P propulsion ",Ef[aantal-1]/time[aantal-1]))
      print("")

   # plotting

   if (doplot==1):
      pyplot.clf()
      ax1 = pyplot.subplot(111)

      pyplot.plot(time, xdot,'r-',label = 'Boat velocity')
      pyplot.plot(time, xdot+ydot,'g-',label = 'Crew velocity')
      pyplot.plot(time, zdot,'b-',label = 'CM velocity')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('v (m/s)')

      ax2 = pyplot.twinx()
      pyplot.plot(time,numpy.degrees(oarangle),'y.',label='oar angle')
      pylab.legend(loc='upper right')
      pyplot.ylabel("Oar Angle (o)")
      ax2.yaxis.tick_right()

      pyplot.show()

   if (doplot==2):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Pf,'r-',label = 'Propulsive power')
      pyplot.plot(time, Pq,'b-',label = 'Kinetic power')
      pyplot.plot(time, Pbladeslip,'k-',label = 'Puddle power')
      pyplot.plot(time, Pf+Pq+Pbladeslip,'g-',label = 'Total power')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()
      
   if (doplot==3):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Ef,'r-',label = 'Propulsive Energy')
      pyplot.plot(time, Eqrower,'b-',label = 'Kinetic Energy')
      pyplot.plot(time, Ef+Eqrower+Eblade,'g-',label = 'Total Energy')
      pyplot.plot(time, Eblade,'k-',label = 'Puddle Energy')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()
      
   if (doplot==4):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Pw,'r-',label = 'Drag sink')
      pyplot.plot(time, Pbladeslip,'k-',label = 'Blade slip sink')
      pyplot.plot(time, Pmb,'b-',label = 'Kinetic energy change boat')
      pyplot.plot(time, Pmc,'g-',label = 'Kinetic energy change crew')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()
      
   if (doplot==5):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Ew+Ediss+Eblade,'r-',label = 'Drag energy + Rower Diss + Blade Slip')
      pyplot.plot(time, Ew, 'y-', label = 'Drag Energy')
      pyplot.plot(time, Ekinb,'b-',label = 'Boat Kinetic energy')
      pyplot.plot(time, Ekinc,'g-',label = 'Crew Kinetic energy')
      pyplot.plot(time, Ew+Ediss+Ekinb+Ekinc+Eblade, 'k-', label = 'Ew + Ediss + Ekinb + Ekinc+Eblade')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()
      
   if (doplot==6):
      pyplot.clf()
      pyplot.subplot(121)
      pyplot.plot(time, Pq,'k-',label = 'Kinetic power')
      pyplot.plot(time, 0*Pq, 'k-')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')

      pyplot.subplot(122)
      pyplot.plot(time, Pqrower,'b-',label = 'Kinetic power rower')
      pyplot.plot(time, Pdiss,'k-',label = 'Kinetic energy dissipation')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')

      pyplot.show()
      
   if (doplot==7):
      pyplot.clf()
      pyplot.plot(time, Ew+Ediss+Ekinb+Ekinc+Eblade, 'r-', label = 'Total Sinks')
      pyplot.plot(time, Ef+Eqrower+Eblade,'g-',label = 'Total Sources')
      pylab.legend(loc='lower right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()

   if (doplot==8):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Phandle,'r-',label = 'Handle power (crew)')
      pyplot.plot(time, Pbladeslip,'g-',label = 'Puddle power')
      pyplot.plot(time, Pf, 'y-', label = 'Propulsive power')
      pyplot.plot(time, Pf+Pbladeslip,'k-',label = 'Propulsive+Puddle Power')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()

   if (doplot==9):
      pyplot.clf()
      ax1 = pyplot.subplot(111)

      pyplot.plot(xblade,yblade,label='blade centre')
      
      pylab.legend(loc='best')
      pyplot.xlabel("x (m)")
      pyplot.ylabel('y (m)')
      ax1.axis('equal')

      xblade2 = xblade[0:len(xblade):4]
      yblade2 = yblade[0:len(xblade):4]
      oarangle2 = oarangle[0:len(xblade):4]

      for i in range(len(xblade2)):
         x1 = xblade2[i]+rigging.bladelength*np.sin(oarangle2[i])/2.
         x2 = xblade2[i]-rigging.bladelength*np.sin(oarangle2[i])/2.
         y1 = yblade2[i]-rigging.bladelength*np.cos(oarangle2[i])/2.
         y2 = yblade2[i]+rigging.bladelength*np.cos(oarangle2[i])/2.

         pyplot.plot([x1,x2],[y1,y2],'r-')

      pyplot.show()

   if (doplot==10):
      pyplot.clf()
      pyplot.plot(time, Fhandle, 'r-', label = 'Handle Force')
      pyplot.plot(time, Fblade, 'g-', label = 'Blade Force')
      pyplot.plot(time, Fprop, 'k-', label = 'Propulsive Force')
      pylab.legend(loc='lower right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('Force (N)')
      pyplot.show()
      

   dv = zdot[len(time)-1]-zdot[0]
   vavg = mean(xdot)
   vend = zdot[len(time)-1]
   energy = max(Ew+Ediss+Eblade-Eloss)
   efficiency = max(Ew-Eloss)/energy
   energy = energy/Nrowers
   power = energy*tempo/60.
   
   return [dv,vend,vavg,ratio,energy,power,efficiency]


def atkinsoncalc(F,crew,rigging,v0=4.3801,dt=0.03,doplot=1,doprint=0,
                  timewise=0,constantdrag=0):
   # calculates one stroke with average handle force as input
   # slide velocity and stroke/recovery ratio are calculated
   # knows about slip, lift, drag. Plots energy balance.


   # initialising output values
   dv = 100.
   vavg = 0.0
   vend = 0.0
   ratio = 0.0
   power = 0.0

  # stroke parameters
   lin = rigging.lin
   lscull = rigging.lscull
   lout = lscull - lin
   tempo = crew.tempo
   mc = crew.mc
   mb = rigging.mb
   recprofile = crew.recprofile
   d = crew.strokelength
   Nrowers = rigging.Nrowers
   try:
       dragform = rigging.dragform
   except:
       dragform = 10.0

   # nr of time steps
   aantal = 1+int(round(60./(tempo*dt)))
   time = linspace(0,60./tempo,aantal)

   vs = zeros(len(time))+v0
   vb = zeros(len(time))+v0
   vc = zeros(len(time))+v0

   oarangle = zeros(len(time))
   xblade = zeros(len(time))
   Fhandle = zeros(len(time))
   Fblade = zeros(len(time))
   Fprop = zeros(len(time))
   Fblade[0:2] = F
   Pbladeslip = zeros(len(time))    # H

   xdotdot = zeros(len(time))
   zdotdot = zeros(len(time))
   ydotdot = zeros(len(time))

   dragconst = zeros(len(time))

   xdot = zeros(len(time))+v0
   ydot = zeros(len(time))+v0
   zdot = zeros(len(time))+v0

   Pf = zeros(len(time))
   Foarlock = zeros(len(time))
   
   handlepos = 0

   # initial handle and boat velocities
   vs[0] = v0
   vb[0] = vs[0]+0.0
   vc[0] = vs[0]+0.0
   oarangle[0] = rigging.oarangle(0)
   xblade[0] = -lout*np.sin(oarangle[0])


   i=1

   vcstroke = 0
   vcstroke2 = 1

   # catch
   vblade = xdot[i-1]
   catchacceler = 1.0*F/mc
   while (vcstroke < vcstroke2):
       vhand = catchacceler*(time[i]-time[0])

       vcstroke = crew.vcm(vhand, handlepos)
       phidot = vb[i-1]*np.cos(oarangle[i-1])
       vhand = phidot*lin*np.cos(oarangle[i-1])
       ydot[i] = vcstroke
       Fdrag = drag_eq((Nrowers*mc)+mb,xdot[i-1],alfaref=alfa*dragform)
       zdotdot[i] = -Fdrag/((Nrowers*mc)+mb)
       zdot[i] = zdot[i-1]+dt*zdotdot[i]
       xdot[i] = zdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[i]

       Fi = crew.forceprofile(F,handlepos)
       Fbladei = Fi*lin/lout 
       res = blade_force(oarangle[i-1],rigging,vb[i-1],Fbladei)
       phidot2 = res[0]
       vhand2 = phidot2*lin*np.cos(oarangle[i-1])
       vcstroke2 = crew.vcm(vhand2,handlepos)

       
       vblade = xdot[i]-phidot*lout*np.cos(oarangle[i-1])
#       print(i,vhand,vhand2,vcstroke,vcstroke2)
       vs[i] = zdot[i]
       vc[i] = xdot[i]+ydot[i]
       vb[i] = xdot[i]

       ydotdot[i] = (ydot[i]-ydot[i-1])/dt
       xdotdot[i] = zdotdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[i]


       handlepos = handlepos+ydot[i]*dt
       Fhandle[i] = 0
       
       oarangle[i] = rigging.oarangle(handlepos)
       i = i+1

   # stroke
   while handlepos<d:
      Fi = crew.forceprofile(F,handlepos)
      Fhandle[i-1] = Fi*lout/lin
      Fblade[i-1] = Fi
      res = blade_force(oarangle[i-1],rigging,vb[i-1],Fblade[i-1])
      phidot = res[0]
#      for u in range(5):
#         l2 = lout-phidot/(vb[i-1]*np.cos(oarangle[i-1]))
#         l2 = lout
#         Fhandle[i-1] = Fblade[i-1]*l2/lin
#         res = blade_force(oarangle[i-1],rigging,vb[i-1],Fblade[i-1])
#         phidot = res[0]

      Fprop[i-1] = res[2]*Nrowers
      phidot = res[0]
      vhand = phidot*lin*np.cos(oarangle[i-1])
#      vcstroke = vhand*(1-(handlepos/d))
      vcstroke = crew.vcm(vhand, handlepos)
      Pbladeslip[i-1] = Nrowers*res[1]*(phidot*lout - vb[i-1]*np.cos(oarangle[i-1]))

      Fdrag = drag_eq((Nrowers*mc)+mb,xdot[i-1],alfaref=alfaatkinson,constantdrag=constantdrag)
      dragconst[i] = Fdrag/(xdot[i-1]**2)
      zdotdot[i] = (Fprop[i-1] - Fdrag)/((Nrowers*mc)+mb)
      zdot[i] = zdot[i-1]+dt*zdotdot[i]
      ydot[i] = vcstroke
      xdot[i] = zdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[i]

      handlepos = handlepos+vhand*dt
      vs[i] = zdot[i]
      vc[i] = xdot[i]+ydot[i]
      vb[i] = xdot[i]

      ydotdot[i] = (ydot[i]-ydot[i-1])/dt
      xdotdot[i] = zdotdot[i]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[i]

      Pf[i-1] = Nrowers*Fblade[i-1]*xdot[i]*np.cos(oarangle[i-1])

      oarangle[i] = rigging.oarangle(handlepos)  

      i = i+1
      
   i=i-1;

   # recovery

   trecovery = max(time)-time[i]
   ratio = time[i]/max(time)

   aantalstroke = i

   if (recprofile == 1): # oude methode (sinus)
      vhandmax = -np.pi*d/(2*trecovery)
      vhand = vhandmax*np.sin(np.pi*(time-time[i])/trecovery)
      for k in range(i+1,aantal):
         Fdrag = drag_eq((Nrowers*mc)+mb,xdot[k-1],alfaref=alfaatkinson,constantdrag=constantdrag)
         dragconst[k] = Fdrag/(xdot[k-1]**2)
         zdotdot[k] = (- Fdrag)/((Nrowers*mc)+mb)
         zdot[k] = zdot[k-1]+dt*zdotdot[k]
         ydot[k] = crew.vcm(vhand[k], handlepos)
         xdot[k] = zdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[k]

         vs[k] = zdot[k]
         vc[k] = xdot[k]+ydot[k]
         vb[k] = xdot[k]

         ydotdot[k] = (ydot[k]-ydot[k-1])/dt
         xdotdot[k] = zdotdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[k]
         handlepos = handlepos+vhand[k]*dt
         oarangle[k] = rigging.oarangle(handlepos)
      
   else:
      vavgrec = d/trecovery
      vcrecovery = zeros(aantal)
      for k in range(i+1,aantal):
         vhand = crew.vhandle(vavgrec,trecovery,time[k]-time[i])
         vcrecovery[k] = crew.vcm(vhand, handlepos)

         Fdrag = drag_eq((Nrowers*mc)+mb,xdot[k-1],alfaref=alfaatkinson,constantdrag=constantdrag)
         dragconst[k] = Fdrag/(xdot[k-1]**2)
         zdotdot[k] = (- Fdrag)/((Nrowers*mc)+mb)
         zdot[k] = zdot[k-1]+dt*zdotdot[k]
         ydot[k] = vcrecovery[k]
         xdot[k] = zdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydot[k]

         vs[k] = zdot[k]
         vc[k] = xdot[k]+ydot[k]
         vb[k] = xdot[k]

         ydotdot[k] = (ydot[k]-ydot[k-1])/dt
         xdotdot[k] = zdotdot[k]-((Nrowers*mc)/((Nrowers*mc)+mb))*ydotdot[k]

         handlepos = d+d*crew.dxhandle(vavgrec,trecovery,time[k]-time[i])
#         handlepos = handlepos+vhand*dt
         oarangle[k] = rigging.oarangle(handlepos)
      

# blade positions      
   xblade=dt*cumsum(vb)-np.sin(oarangle)*lout
   yblade=lout*np.cos(oarangle)+rigging.spread


# velocities
   xdot = vb
   zdot = vs
   ydot = vc-vb

   xdotdot[1]=(xdot[1]-xdot[0])/dt
   ydotdot[1]=(ydot[1]-ydot[0])/dt

   Pq = (Nrowers*mc)*(xdotdot+ydotdot)*ydot
   
#   Ekinb = 0.5*mb*xdot**2 - 0.5*mb*v0**2
#   Ekinc = 0.5*mc*(xdot+ydot)**2 - 0.5*mc*v0**2

   Pw = drag_eq((Nrowers*mc)+mb,xdot,alfaref=alfaatkinson,constantdrag=constantdrag)*xdot

   Pmb = mb*xdot*xdotdot
   Pmc = (Nrowers*mc)*(xdot+ydot)*(xdotdot+ydotdot)

   Phandle = Nrowers*Fhandle*(xdot+ydot)*np.cos(oarangle)

   Pleg = Nrowers*mc*(xdotdot+ydotdot)*ydot

   Ekinb = cumsum(Pmb)*dt
   Ekinc = cumsum(Pmc)*dt

   Pqrower = abs(Pq)
   Pdiss = Pqrower-Pq

   Ef = cumsum(Pf)*dt
   Eq = cumsum(Pq)*dt
   Eblade = cumsum(Pbladeslip)*dt
   Eqrower = cumsum(Pqrower)*dt
   Ediss = cumsum(Pdiss)*dt
   Ew = cumsum(Pw)*dt
   Eleg = cumsum(Pleg)*dt
   Ehandle = cumsum(Phandle)*dt

   # printing
   if (doprint==1):
      print(("E blade ",Eblade[aantal-1]))
      print(("Ediss rower ",Ediss[aantal-1]))
      print(("E drag ",Ew[aantal-1]))
      print(("Eleg ",Eleg[aantal-1]))
      print(("Ehandle ",Ehandle[aantal-1]))
      print(("Epropulsion ",Ef[aantal-1]))
      print("")
      print(("P blade ",Eblade[aantal-1]/time[aantal-1]))
      print(("P momentum ",Eqrower[aantal-1]/time[aantal-1]))
      print(("P dissipation ",Ediss[aantal-1]/time[aantal-1]))
      print(("P leg   ",Eleg[aantal-1]/time[aantal-1]))
      print(("P handle ",Ehandle[aantal-1]/time[aantal-1]))
      print(("P drag ",Ew[aantal-1]/time[aantal-1]))
      print(("P propulsion ",Ef[aantal-1]/time[aantal-1]))
      print("")

   # plotting

   if (doplot==1):
      pyplot.clf()
      ax1 = pyplot.subplot(111)

      pyplot.plot(time, xdot,'r-',label = 'Boat velocity')
      pyplot.plot(time, xdot+ydot,'g-',label = 'Crew velocity')
      pyplot.plot(time, zdot,'b-',label = 'CM velocity')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('v (m/s)')

      ax2 = pyplot.twinx()
      pyplot.plot(time,numpy.degrees(oarangle),'y.',label='oar angle')
      pylab.legend(loc='upper right')
      pyplot.ylabel("Oar Angle (o)")
      ax2.yaxis.tick_right()

      pyplot.show()

   if (doplot==2):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Pf,'r-',label = 'Propulsive power')
      pyplot.plot(time, Pq,'b-',label = 'Kinetic power')
      pyplot.plot(time, Pbladeslip,'k-',label = 'Puddle power')
      pyplot.plot(time, Pf+Pq+Pbladeslip,'g-',label = 'Total power')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()
      
   if (doplot==3):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Ef,'r-',label = 'Propulsive Energy')
      pyplot.plot(time, Eqrower,'b-',label = 'Kinetic Energy')
      pyplot.plot(time, Ef+Eqrower+Eblade,'g-',label = 'Total Energy')
      pyplot.plot(time, Eblade,'k-',label = 'Puddle Energy')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()
      
   if (doplot==4):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Pw,'r-',label = 'Drag sink')
      pyplot.plot(time, Pbladeslip,'k-',label = 'Blade slip sink')
      pyplot.plot(time, Pmb,'b-',label = 'Kinetic energy change boat')
      pyplot.plot(time, Pmc,'g-',label = 'Kinetic energy change crew')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()
      
   if (doplot==5):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Ew+Ediss+Eblade,'r-',label = 'Drag energy + Rower Diss + Blade Slip')
      pyplot.plot(time, Ew, 'y-', label = 'Drag Energy')
      pyplot.plot(time, Ekinb,'b-',label = 'Boat Kinetic energy')
      pyplot.plot(time, Ekinc,'g-',label = 'Crew Kinetic energy')
      pyplot.plot(time, Ew+Ediss+Ekinb+Ekinc+Eblade, 'k-', label = 'Ew + Ediss + Ekinb + Ekinc+Eblade')
      pylab.legend(loc='upper left')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()
      
   if (doplot==6):
      pyplot.clf()
      pyplot.subplot(121)
      pyplot.plot(time, Pq,'k-',label = 'Kinetic power')
      pyplot.plot(time, 0*Pq, 'k-')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')

      pyplot.subplot(122)
      pyplot.plot(time, Pqrower,'b-',label = 'Kinetic power rower')
      pyplot.plot(time, Pdiss,'k-',label = 'Kinetic energy dissipation')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')

      pyplot.show()
      
   if (doplot==7):
      pyplot.clf()
      pyplot.plot(time, Ew+Ediss+Ekinb+Ekinc+Eblade, 'r-', label = 'Total Sinks')
      pyplot.plot(time, Ef+Eqrower+Eblade,'g-',label = 'Total Sources')
      pylab.legend(loc='lower right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('energy (J)')
      pyplot.show()

   if (doplot==8):
      pyplot.clf()
      pyplot.subplot(111)
      pyplot.plot(time, Phandle,'r-',label = 'Handle power (crew)')
      pyplot.plot(time, Pbladeslip,'g-',label = 'Puddle power')
      pyplot.plot(time, Pf, 'y-', label = 'Propulsive power')
      pyplot.plot(time, Pf+Pbladeslip,'k-',label = 'Propulsive+Puddle Power')
      pylab.legend(loc='upper right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('power (W)')
      pyplot.show()

   if (doplot==9):
      pyplot.clf()
      ax1 = pyplot.subplot(111)

      pyplot.plot(xblade,yblade,label='blade centre')
      
      pylab.legend(loc='best')
      pyplot.xlabel("x (m)")
      pyplot.ylabel('y (m)')
      ax1.axis('equal')

      xblade2 = xblade[0:len(xblade):4]
      yblade2 = yblade[0:len(xblade):4]
      oarangle2 = oarangle[0:len(xblade):4]

#      for i in range(len(xblade2)):
#         x1 = xblade2[i]+rigging.bladelength*np.sin(oarangle2[i])/2.
#         x2 = xblade2[i]-rigging.bladelength*np.sin(oarangle2[i])/2.
#         y1 = yblade2[i]-rigging.bladelength*np.cos(oarangle2[i])/2.
#         y2 = yblade2[i]+rigging.bladelength*np.cos(oarangle2[i])/2.

#         pyplot.plot([x1,x2],[y1,y2],'r-')

      pyplot.show()

   if (doplot==10):
      pyplot.clf()
      pyplot.plot(time, Fhandle, 'r-', label = 'Handle Force')
      pyplot.plot(time, Fblade, 'g-', label = 'Blade Force')
      pyplot.plot(time, Fprop, 'k-', label = 'Propulsive Force')
      pylab.legend(loc='lower right')
      pyplot.xlabel("time (s)")
      pyplot.ylabel('Force (N)')
      pyplot.show()

   if (doplot==11):
      pyplot.clf()
      pyplot.plot(time,dragconst)
      pyplot.xlabel("time (s)")
      pyplot.ylabel("Drag constant (N s^2 / m^2)")
      pyplot.show()
      print(("average K :",average(dragconst)))

   dv = zdot[len(time)-1]-zdot[0]
   vavg = mean(xdot)
   vend = zdot[len(time)-1]
   energy = max(Ew+Ediss+Eblade)
   efficiency = max(Ew)/energy
   energy = energy/Nrowers
   power = energy*tempo/60.
   vmin = min(xdot)
   vmax = max(xdot)
   
   return [dv,vend,vavg,ratio,energy,power,efficiency,vmin,vmax]



def stroke(F,crew,rigging,v0,dt,aantal,doplot=0,timewise=0,catchacceler=5,
           dowind=1,windv=0):
    """ Calculates a few (aantal) strokes and returns parameters averaged
    over those strokes

    """
    
    dv=0
    vend=0
    vavg=0
    vmin = 0
    vmax = 0
    ratio=0
    energy=0
    power=0
    eff=0
    cn_check = 0
    RIM_E = 0
    RIM_check = 0
    RIM_catchE = 0
    RIM_catchD = 0
    drag_eff = 0
    efficiency = 0
    tcatchacceler = catchacceler

    for i in range(aantal):
        res = energybalance(F,crew,rigging,v0,dt,0,timewise=timewise,
                            catchacceler=tcatchacceler,
                            dowind=dowind,windv=windv)
        dv = dv+res[0]
        vend = vend+res[1]
        vavg = vavg + res[2]
        ratio = ratio + res[3]
        energy = energy+res[4]
        power = power+res[5]
        vmin = vmin+res[8]
        vmax = vmax+res[7]
        eff=eff+res[6]
        v0 = res[1]
        cn_check = cn_check+res[9]
        RIM_E = RIM_E+res[10]
        RIM_check = RIM_check+res[11]
        RIM_catchE = RIM_catchE+res[12]
        RIM_catchD = RIM_catchD+res[13]
        catchacceler = catchacceler+res[14]
        drag_eff = drag_eff+res[15]
        tcatchacceler = res[14]


    dv = dv/aantal
    vend = vend/aantal
    vavg = vavg/aantal
    vmin = vmin/aantal
    vmax = vmax/aantal
    ratio = ratio/aantal
    energy = energy/aantal
    power = power/aantal
    eff=eff/aantal
    cn_check=cn_check/aantal
    RIM_E = RIM_E/aantal
    RIM_check = RIM_check/aantal
    RIM_catchE = RIM_catchE/aantal
    RIM_catchD = RIM_catchD/aantal
    drag_eff = drag_eff/aantal
    catchacceler = catchacceler/aantal

    if (doplot):
        res = energybalance(F,crew,rigging,vend,dt,doplot,timewise=timewise,
                            catchacceler=catchacceler,
                            dowind=dowind,windv=windv)

    return [dv,vend,vavg,ratio,energy,power,eff,vmax,vmin,cn_check,RIM_E,RIM_check,RIM_catchE,RIM_catchD,catchacceler,drag_eff]



def stroke_erg(ratio,crew,erg,w0,dt,aantal,doplot=0,theconst=0.0):
    """ Calculates a number (aantal) erg strokes and returns
    parameters averaged over the strokes
    """
    
    dv=0
    wend=0
    wavg=0

    energy=0
    power=0
    powerd = 0
    eff=0
    tcatchacceler=5
    catchacceler=0

    for i in range(aantal):
        res = energybalance_erg(ratio,crew,erg,w0,dt,0,theconst=theconst)
        dv = dv+res[0]

        wend = wend+res[1]
        wavg = wavg + res[2]

        energy = energy+res[4]
        power = power+res[5]
        powerd = powerd+res[6]
        w0 = res[1]

    dv = dv/aantal
    wend = wend/aantal
    wavg = wavg/aantal
    

    energy = energy/aantal
    power = power/aantal
    powerd = powerd/aantal

    if (doplot):
        res = energybalance_erg(ratio,crew,erg,wend,dt,doplot,theconst=theconst)

    return [dv,wend,wavg,ratio,energy,power,powerd]

def stroke_atkinson(F,crew,rigging,v0,dt,aantal,timewise=0,constantdrag=0):
   dv=0
   vend=0
   vavg=0
   ratio=0
   energy=0
   power=0
   eff=0

   for i in range(aantal):
      res = atkinsoncalc(F,crew,rigging,v0,dt,0,timewise=timewise,constantdrag=constantdrag)
      dv = dv+res[0]
      vend = vend+res[1]
      vavg = vavg + res[2]
      ratio = ratio + res[3]
      energy = energy+res[4]
      power = power+res[5]
      eff=eff+res[6]
      v0 = res[1]

   dv = dv/aantal
   vend = vend/aantal
   vavg = vavg/aantal
   ratio = ratio/aantal
   energy = energy/aantal
   power = power/aantal
   eff=eff/aantal

   return [dv,vend,vavg,ratio,energy,power,eff]

def constantwatt(watt,crew,rigging,timestep=0.03,aantal=5,
                 aantal2=5,Fmin=50,Fmax=1000,catchacceler=5,
                 windv=0,dowind=1):
    """ Returns force, average speed given an input power in watt
    """

    F = linspace(Fmin,Fmax,aantal)
    velocity = zeros(aantal)
    power = zeros(aantal)
    ratios = zeros(aantal)
    energies = zeros(aantal)
    tcatchacceler = catchacceler
   
    for i in range(len(F)):
        # een paar halen om op snelheid te komen
        dv = 1
        vend = 4.0
        while (dv/vend > 0.001):
            res = energybalance(F[i],crew,rigging,vend,timestep,0,
                                catchacceler=tcatchacceler,
                                windv=windv,dowind=dowind)
            dv = res[0]
            vend = res[1]
            tcatchacceler = res[14]
        res = stroke(F[i],crew,rigging,vend,timestep,10,
                     dowind=dowind,windv=windv)
        velocity[i] = res[2]
        ratios[i] = res[3]
        energies[i] = res[4]
        power[i] = res[5]


    fres = sr_interpol1(F,power,watt)

    Fmin = fres-10
    Fmax = fres+10

    F = linspace(Fmin,Fmax,aantal2)

    velocity = zeros(aantal2)
    power = zeros(aantal2)
    ratios = zeros(aantal2)
    energies = zeros(aantal2)

    for i in range(len(F)):
        # een paar halen om op snelheid te komen
        dv = 1
        vend = 4.0
        while (dv/vend > 0.001):
            res = energybalance(F[i],crew,rigging,vend,
                                timestep,0,catchacceler=tcatchacceler,
                                dowind=dowind,windv=windv)
            dv = res[0]
            vend = res[1]
            tcatchacceler = res[14]
        res = stroke(F[i],crew,rigging,vend,timestep,10,
                     dowind=dowind,windv=windv)
        velocity[i] = res[2]
        ratios[i] = res[3]
        energies[i] = res[4]
        power[i] = res[5]

   
    fres = sr_interpol1(F,power,watt)

    while (dv/vend > 0.001):
        res = energybalance(fres,crew,rigging,vend,timestep,0,
                            catchacceler=tcatchacceler,
                            dowind=dowind,windv=windv)
        dv = res[0]
        vend = res[1]
        tcatchacceler = res[14]

   
    res = stroke(fres,crew,rigging,vend,timestep,10,catchacceler=tcatchacceler,
                 dowind=dowind,windv=windv)
    vavg = res[2]
    ratio = res[3]
    pw = res[5]
    eff = res[6]


    return [fres,vavg,ratio,pw,eff]

def constantwattfast(watt,crew,rigging,timestep=0.03,aantal=5,
                 aantal2=5,Fmin=50,Fmax=1000,catchacceler=5,
                     windv=0,dowind=1, max_iterations_allowed=15):
    """ Returns force, average speed given an input power in watt
    """

    F = linspace(Fmin,Fmax,aantal)
    velocity = zeros(aantal)
    power = zeros(aantal)
    ratios = zeros(aantal)
    energies = zeros(aantal)
    tcatchacceler = catchacceler
   
    for i in range(len(F)):
        # een paar halen om op snelheid te komen
        dv = 1
        vend = 4.0
        count = 0
        while (dv/vend > 0.001) and count < max_iterations_allowed:
            res = energybalance(F[i],crew,rigging,vend,timestep,0,
                                catchacceler=tcatchacceler,
                                windv=windv,dowind=dowind)
            dv = res[0]
            vend = res[1]
            tcatchacceler = res[14]
            count += 1
            
        res = stroke(F[i],crew,rigging,vend,timestep,10,
                     dowind=dowind,windv=windv)
        velocity[i] = res[2]
        ratios[i] = res[3]
        energies[i] = res[4]
        power[i] = res[5]


    fres = sr_interpol1(F,power,watt)
    count = 0

    while (dv/vend > 0.001) and count < max_iterations_allowed:
        res = energybalance(fres,crew,rigging,vend,timestep,0,
                            catchacceler=tcatchacceler,
                            dowind=dowind,windv=windv)
        dv = res[0]
        vend = res[1]
        tcatchacceler = res[14]
        count += 1

   
    res = stroke(fres,crew,rigging,vend,timestep,10,catchacceler=tcatchacceler,
                 dowind=dowind,windv=windv)
    vavg = res[2]
    ratio = res[3]
    pw = res[5]
    eff = res[6]


    return [fres,vavg,ratio,pw,eff]

def constantwatt_erg(watt,crew,erg,timestep=0.03,aantal=5,
                     aantal2=5,ratiomin=0.4,ratiomax=0.6,theconst=1.0):
   """ Returns drive/recovery ratio, force given an input power (watt)

   The power is the total power (not only what the erg display shows)
   
   """

   F = linspace(ratiomin,ratiomax,aantal)
   velocity = zeros(aantal)
   power = zeros(aantal)
   powerd = zeros(aantal)
   ratios = zeros(aantal)
   energies = zeros(aantal)
   
   for i in range(len(F)):
       # een paar halen om op snelheid te komen

       dv = 1
       vend = 40.
       while (dv/vend > 0.001):
           res = energybalance_erg(F[i],crew,erg,vend,timestep,0,theconst=theconst)
           dv = res[0]
           vend = res[1]

       res = stroke_erg(F[i],crew,erg,vend,timestep,10,theconst=theconst)
       velocity[i] = res[2]
       ratios[i] = res[3]
       energies[i] = res[4]
       power[i] = res[5]
       powerd[i] = res[6]

       

#   fres = sr_interpol3(F,power,watt)
   fres = sr_interpol4(F,power,watt)

   ratiomin = fres-0.1
   ratiomax = fres+0.1

   F = linspace(ratiomin,ratiomax,aantal2)

   velocity = zeros(aantal2)
   power = zeros(aantal2)
   powerd = zeros(aantal2)
   ratios = zeros(aantal2)
   energies = zeros(aantal2)

   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 40.
      while (dv/vend > 0.001):
          try:
              res = energybalance_erg(F[i],crew,erg,vend,timestep,0,theconst=theconst)
              dv = res[0]
              vend = res[1]
          except:
              pass
          
      res = stroke_erg(F[i],crew,erg,vend,timestep,10,theconst=theconst)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]
      powerd[i] = res[6]
   

#   fres = sr_interpol3(F,power,watt)
   fres = sr_interpol4(F,power,watt)
   
   while (dv/vend > 0.001):
      res = energybalance_erg(fres,crew,erg,vend,timestep,0,theconst=theconst)
   
   res = stroke_erg(fres,crew,erg,vend,timestep,10,theconst=theconst)
   vavg = res[2]
   ratio = res[3]
   pw = res[5]
   pwd = res[6]

   return [fres,vavg,ratio,pw,pwd]

def constantwatt_ergtempo(watt,crew,erg,timestep=0.03,aantal=5,
                     aantal2=5,tempomin=15,tempomax=45,theconst=1.0,ratio=0.5):
   """ Returns drive/recovery ratio, force given an input power (watt)

   The power is the total power (not only what the erg display shows)
   
   """

   F = linspace(tempomin,tempomax,aantal)
   velocity = zeros(aantal)
   power = zeros(aantal)
   powerd = zeros(aantal)
   ratios = zeros(aantal)
   energies = zeros(aantal)
   
   for i in range(len(F)):
       # een paar halen om op snelheid te komen

       dv = 1
       vend = 40.
       crew.tempo = F[i]
       while (dv/vend > 0.001):
           res = energybalance_erg(ratio,crew,erg,vend,timestep,0,theconst=theconst)
           dv = res[0]
           vend = res[1]

       res = stroke_erg(ratio,crew,erg,vend,timestep,10,theconst=theconst)
       velocity[i] = res[2]
       ratios[i] = res[3]
       energies[i] = res[4]
       power[i] = res[5]
       powerd[i] = res[6]

       

   fres = sr_interpol3(F,power,watt)

   tempomin = fres-1.0
   tempomax = fres+1.0

   F = linspace(tempomin,tempomax,aantal2)

   velocity = zeros(aantal2)
   power = zeros(aantal2)
   powerd = zeros(aantal2)
   ratios = zeros(aantal2)
   energies = zeros(aantal2)

   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 40.
      crew.tempo = F[i]
      while (dv/vend > 0.001):
          try:
              res = energybalance_erg(ratio,crew,erg,vend,timestep,0,theconst=theconst)
              dv = res[0]
              vend = res[1]
          except:
              pass
          
      res = stroke_erg(ratio,crew,erg,vend,timestep,10,theconst=theconst)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]
      powerd[i] = res[6]
   

   fres = sr_interpol3(F,power,watt)
   crew.tempo = fres

   while (dv/vend > 0.001):
      res = energybalance_erg(ratio,crew,erg,vend,timestep,0,theconst=theconst)
   
   res = stroke_erg(ratio,crew,erg,vend,timestep,10,theconst=theconst)
   vavg = res[2]
   ratio = res[3]
   pw = res[5]
   pwd = res[6]

   return [fres,vavg,ratio,pw,pwd]

def find_ergpower(watt,crew,erg,timestep=0.03,aantal=10,aantal2=10,powermin=100,powermax=400,ratio=0.5):
    F = linspace(powermin,powermax,aantal)
    power = zeros(aantal)

    for i in range(len(F)):
        res = ergpowertopower(F[i],ratio,crew,erg,aantal=5,aantal2=5)
        power[i] = res[0]

    fres = sr_interpol1(F,power,watt)

    powermin = 0.9*fres
    powermax = 1.1*fres

    F = linspace(powermin,powermax,aantal2)

    for i in range(len(F)):
        res = ergpowertopower(F[i],ratio,crew,erg,aantal=5,aantal2=5)
        power[i] = res[0]

    fres = sr_interpol1(F,power,watt)
    if fres<0:
        fres = 50.

    return fres

def constantwatt_ergdisplay(watt,crew,erg,timestep=0.03,aantal=10,
                            aantal2=10,ratiomin=0.3,ratiomax=0.8,theconst=0.0,
                            catchacceler=5):
   """ Returns drive/recovery ratio given an input power (watt)

   """


   F = linspace(ratiomin,ratiomax,aantal)
   velocity = zeros(aantal)
   power = zeros(aantal)
   powerd = zeros(aantal)
   ratios = zeros(aantal)
   energies = zeros(aantal)
   
   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 40.0
      while (dv/vend > 0.001):
         res = energybalance_erg(F[i],crew,erg,vend,timestep,0,theconst=theconst)
         dv = res[0]
         vend = res[1]
      res = stroke_erg(F[i],crew,erg,vend,timestep,10,theconst=theconst)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]
      powerd[i] = res[6]



   fres = sr_interpol1(F,powerd,watt)

   ratiomin = fres-0.1
   ratiomax = fres+0.1

   F = linspace(ratiomin,ratiomax,aantal2)

   velocity = zeros(aantal2)
   power = zeros(aantal2)
   powerd = zeros(aantal2)
   ratios = zeros(aantal2)
   energies = zeros(aantal2)

   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 40.
      while (dv/vend > 0.001):
         res = energybalance_erg(F[i],crew,erg,vend,timestep,0,theconst=theconst)
         dv = res[0]
         vend = res[1]
      res = stroke_erg(F[i],crew,erg,vend,timestep,10,theconst=theconst)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]
      powerd[i] = res[6]

   fres = sr_interpol1(F,powerd,watt)

   while (dv/vend > 0.001):
      res = energybalance_erg(fres,crew,erg,vend,timestep,0,theconst=theconst)
   
   res = stroke_erg(fres,crew,erg,vend,timestep,10,theconst=theconst)
   vavg = res[2]
   ratio = res[3]
   pw = res[5]
   pwd = res[6]

   return [fres,vavg,ratio,pw,pwd]

def constantvelo(velo,crew,rigging,timestep=0.03,aantal=5,
                 aantal2=5,Fmin=100,Fmax=400,catchacceler=5,dowind=1,windv=0):
   """ Returns the force and power needed to achieve
   average boat speed of velo

   """
   
   F = linspace(Fmin,Fmax,aantal)
   velocity = zeros(aantal)
   power = zeros(aantal)
   ratios = zeros(aantal)
   energies = zeros(aantal)
   tcatchacceler=catchacceler
   
   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 4.0
      while (dv/vend > 0.001):
         res = energybalance(F[i],crew,rigging,vend,timestep,0,
                             catchacceler=tcatchacceler,
                             windv=windv,dowind=dowind)
         dv = res[0]
         vend = res[1]
         tcatchacceler = res[14]
      res = stroke(F[i],crew,rigging,vend,timestep,10,
                   windv=windv,dowind=dowind)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]


   fres = sr_interpol1(F,velocity,velo)

   Fmin = fres-10
   Fmax = fres+10

   F = linspace(Fmin,Fmax,aantal2)

   velocity = zeros(aantal2)
   power = zeros(aantal2)
   ratios = zeros(aantal2)
   energies = zeros(aantal2)

   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 4.0
      while (dv/vend > 0.001):
         res = energybalance(F[i],crew,rigging,vend,
                             timestep,0,catchacceler=tcatchacceler,
                             dowind=dowind,windv=windv)
         dv = res[0]
         vend = res[1]
         tcatchacceler = res[14]
      res = stroke(F[i],crew,rigging,vend,timestep,10,
                   dowind=dowind,windv=windv)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]
   
   fres = sr_interpol1(F,velocity,velo)

   while (dv/vend > 0.001):
      res = energybalance(fres,crew,rigging,vend,timestep,0,
                          catchacceler=tcatchacceler,
                          dowind=dowind,windv=windv)
      vend = res[1]
      tcatchacceler = res[14]
      dv=res[0]

   
   res = stroke(fres,crew,rigging,vend,timestep,10,
                windv=windv,dowind=dowind)
   vavg = res[2]
   ratio = res[3]
   pw = res[5]
   eff = res[6]

   return [fres,vavg,ratio,pw,eff]

def constantvelofast(velo,crew,rigging,timestep=0.03,aantal=5,
                 aantal2=5,Fmin=100,Fmax=400,catchacceler=5,
                     windv=0,dowind=1):
   """ Returns the force and power needed to achieve
   average boat speed of velo

   Cuts a few corners to speed up the calculation

   """
   
   F = linspace(Fmin,Fmax,aantal)
   velocity = zeros(aantal)
   power = zeros(aantal)
   ratios = zeros(aantal)
   energies = zeros(aantal)
   tcatchacceler=catchacceler
   
   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 4.
      while (dv/vend > 0.001):
          res = energybalance(F[i],crew,rigging,vend,timestep,0,
                              catchacceler=tcatchacceler,
                              dowind=dowind,windv=windv)
          dv = res[0]
          vend = res[1]
          tcatchacceler = res[14]
      res = stroke(F[i],crew,rigging,vend,timestep,10,
                   dowind=dowind,windv=windv)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]


   fres = sr_interpol1(F,velocity,velo)


   while (dv/vend > 0.001):
      res = energybalance(fres,crew,rigging,vend,timestep,0,
                          catchacceler=tcatchacceler,
                          dowind=dowind,windv=windv)
      vend = res[1]
      tcatchacceler = res[14]
      dv=res[0]

   
   res = stroke(fres,crew,rigging,vend,timestep,10,
                dowind=dowind,windv=windv)
   vavg = res[2]
   ratio = res[3]
   pw = res[5]
   eff = res[6]

   return [fres,vavg,ratio,pw,eff]


def constantratio(ratio,crew,rigging,timestep=0.03,aantal=5,
                  aantal2=5,Fmin=100,Fmax=400,catchacceler=5,
                  windv=0,dowind=1):
   """ Finds the force, power and speed needed to achieve a
   certain drive/recovery ratio

   """
  
   F = linspace(Fmin,Fmax,aantal)
   velocity = zeros(aantal)
   power = zeros(aantal)
   ratios = zeros(aantal)
   energies = zeros(aantal)
   tcatchacceler=catchacceler
   
   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 4.0
      while (dv/vend > 0.001):
         res = energybalance(F[i],crew,rigging,vend,timestep,0,
                             catchacceler=tcatchacceler,
                             windv=windv,dowind=dowind)
         dv = res[0]
         vend = res[1]
         tcatchacceler = res[14]
      res = stroke(F[i],crew,rigging,vend,timestep,10,
                   dowind=dowind,windv=windv)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]


   fres = sr_interpol1(F,ratios,ratio)

   Fmin = fres-10
   Fmax = fres+10

   F = linspace(Fmin,Fmax,aantal2)

   velocity = zeros(aantal2)
   power = zeros(aantal2)
   ratios = zeros(aantal2)
   energies = zeros(aantal2)

   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 4.0
      while (dv/vend > 0.001):
         res = energybalance(F[i],crew,rigging,vend,timestep,0,
                             catchacceler=tcatchacceler,
                             windv=windv,dowind=dowind)
         dv = res[0]
         vend = res[1]
         tcatchacceler = res[14]
      res = stroke(F[i],crew,rigging,vend,timestep,10,
                   windv=windv,dowind=dowind)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]
   
   fres = sr_interpol1(F,ratios,ratio)

   while (dv/vend > 0.001):
      res = energybalance(fres,crew,rigging,vend,timestep,0,
                          catchacceler=tcatchacceler,
                          windv=windv,dowind=dowind)
      vend = res[1]
      tcatchacceler = res[14]
      dv=res[0]

   
   res = stroke(fres,crew,rigging,vend,timestep,10,
                windv=windv,dowind=dowind)
   vavg = res[2]
   ratio = res[3]
   pw = res[5]
   eff = res[6]

   return [fres,vavg,ratio,pw,eff]

def constantrecovery(trecovery,crew,rigging,timestep=0.03,
                     aantal=5,aantal2=5,Fmin=100,Fmax=400,\
                     windv=0,dowind=1):
   """ Finds the force, power and average boat speed to row a given stroke
   rate with a recovery duration of trecovery seconds
   """

   F = linspace(Fmin,Fmax,aantal)
   velocity = zeros(aantal)
   power = zeros(aantal)
   ratios = zeros(aantal)
   energies = zeros(aantal)
   
   tstroke = 60./crew.tempo
   tratio = (tstroke-trecovery)/tstroke

   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 4.0
      while (dv/vend > 0.001):
         res = energybalance(F[i],crew,rigging,vend,timestep,0,
                             dowind=dowind,windv=windv)
         dv = res[0]
         vend = res[1]
      res = stroke(F[i],crew,rigging,vend,timestep,10,
                   dowind=dowind,windv=windv)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]

   fres = sr_interpol1(F,ratios,tratio)

   Fmin = fres-10
   Fmax = fres+10

   F = linspace(Fmin,Fmax,aantal2)

   velocity = zeros(aantal2)
   power = zeros(aantal2)
   ratios = zeros(aantal2)
   energies = zeros(aantal2)

   for i in range(len(F)):
      # een paar halen om op snelheid te komen
      dv = 1
      vend = 4.0
      while (dv/vend > 0.001):
         res = energybalance(F[i],crew,rigging,vend,timestep,0,
                             dowind=dowind,windv=windv)
         dv = res[0]
         vend = res[1]
      res = stroke(F[i],crew,rigging,vend,timestep,10,
                   dowind=dowind,windv=windv)
      velocity[i] = res[2]
      ratios[i] = res[3]
      energies[i] = res[4]
      power[i] = res[5]
   
   fres = sr_interpol1(F,ratios,tratio)

   while (dv/vend > 0.001):
      res = energybalance(fres,crew,rigging,vend,timestep,0,
                          windv=windv,dowind=dowind)
   
   res = stroke(fres,crew,rigging,vend,timestep,10,
                windv=windv,dowind=dowind)
   vavg = res[2]
   ratio = res[3]
   trecovery = (1-ratio)*tstroke
   pw = res[5]
   eff = res[6]
   vend  = res[1] 

   print(("vend",vend,res[1]))

   return [fres,trecovery,vavg,vend,ratio,pw,eff]


def drag_skif():
    """ Plots the drag of a single as a function of boat speed
    """
    
    velo = linspace(0,8,100)

    crewweight = 80.0
    boatweight = 14.0

    displacement = crewweight+boatweight
    
    a2 = 3.5

    W1 = drag_eq(displacement,velo,alfaref=alfa*dragform)
    W2 = a2*velo**2

    pyplot.clf()
    pyplot.plot(velo,W1,label='ITTC 1957')
    pyplot.plot(velo,W2,label='Constant alpha')
    pylab.legend(loc='best')
    pyplot.xlabel("Boat velocity (m/s)")
    pyplot.ylabel("Drag Force (N)")

    pyplot.show()

    

    return 1

def drag_eight():
    """ Plots the drag force of an eight
    """
    
    velo = linspace(0,8,100)

    crewweight = 8*80.0
    boatweight = 98.0

    displacement = crewweight+boatweight
    
    a2 = 3.5
    a_eight = a2*(displacement/(94.0))**(2./3.)

    W1 = drag_eq(displacement,velo,alfaref=alfa*dragform)
    W2 = a_eight*velo**2

    pyplot.clf()
    pyplot.plot(velo,W1,label='ITTC 1957')
    pyplot.plot(velo,W2,label='Constant alpha')
    pylab.legend(loc='best')
    pyplot.xlabel("Boat velocity (m/s)")
    pyplot.ylabel("Drag Force (N)")

    pyplot.show()

    

    return 1

def drag_pair():
    """ Plots the drag force of a pair
    """
    
    velo = linspace(0,8,100)

    crewweight = 2*80.0
    boatweight = 27.0

    displacement = crewweight+boatweight
    
    a2 = 3.5
    a_pair = a2*(displacement/(94.0))**(2./3.)

    W1 = drag_eq(displacement,velo,alfaref=alfa*dragform)
    W2 = a_pair*velo**2

    pyplot.clf()
    pyplot.plot(velo,W1,label='ITTC 1957')
    pyplot.plot(velo,W2,label='Constant alpha')
    pylab.legend(loc='best')
    pyplot.xlabel("Boat velocity (m/s)")
    pyplot.ylabel("Drag Force (N)")

    pyplot.show()

    

    return 1

def powertoerg(pw,ratio,crew,erg):
   """ Returns erg power, kinetic power and erg split for a given input power
   """
#   tempo = crew.tempo
#   mc = crew.mc
#   strokelength = crew.strokelength
#   cmdistance = strokelength/2.0

#   timetotal = 60./tempo
#   timestroke = ratio*timetotal
#   timerecovery = (1-ratio)*timetotal

#   vavgstroke = cmdistance/timestroke
#   vmaxstroke = np.pi*vavgstroke/2.
#   Emaxstroke = 0.5*mc*vmaxstroke**2.0

#   vavgrecovery = cmdistance/timerecovery
#   vmaxrecovery = np.pi*vavgrecovery/2.
#   Emaxrecovery = 0.5*mc*vmaxrecovery**2.0

#   kinpower = (Emaxstroke+Emaxrecovery)/timetotal
#   ergpower = pw-kinpower

   result = constantwatt_erg(pw,crew,erg)
   
   ergpower = result[4]
   kinpower = pw-ergpower

   ergvelo = (ergpower/2.8)**(1./3.)
   
   ergminsec = vavgto500mtime(ergvelo)

   ergsplit = str(int(round(ergminsec[0])))+':'+str(int(round(ergminsec[1])))

   return [ergpower,kinpower,ergvelo,ergsplit]

def ergpowertopower(ergpower,ratio,crew,erg,aantal=5,aantal2=5):
   result = constantwatt_ergdisplay(ergpower,crew,erg,theconst=1.0,aantal=aantal,aantal2=aantal2)
   totalpower = result[3]
   kinpower = totalpower-ergpower

   return [totalpower,ergpower,kinpower]

def ergtopower(min,sec,ratio,crew,erg):
   """ Returns total power, erg display power and kinetic power given a split in min, sec
   """
#   tempo = crew.tempo
#   mc = crew.mc
#   strokelength = crew.strokelength
#   cmdistance = strokelength/2.0

#   timetotal = 60./tempo
#   timestroke = ratio*timetotal
#   timerecovery = (1-ratio)*timetotal

#   vavgstroke = cmdistance/timestroke
#   vmaxstroke = np.pi*vavgstroke/2.
#   Emaxstroke = 0.5*mc*vmaxstroke**2.0

#   vavgrecovery = cmdistance/timerecovery
#   vmaxrecovery = np.pi*vavgrecovery/2.
#   Emaxrecovery = 0.5*mc*vmaxrecovery**2.0

#   kinpower = (Emaxstroke+Emaxrecovery)/timetotal
   velo = 500./(60.*min+sec)
   ergpower = 2.8*velo**3
   
   result = constantwatt_ergdisplay(ergpower,crew,erg,theconst=1.0)
   totalpower = result[3]
   kinpower = totalpower-ergpower

   return [totalpower,ergpower,kinpower]

def ergtoboatspeed(min,sec,ratio,crew,rigging,erg):
   """ Calculates boat speed, given an erg split for given crew, boat, erg
   """
   res = ergtopower(min,sec,ratio,crew,erg)
   pw = res[0]
   
   res = constantwatt(pw,crew,rigging)

   return res
