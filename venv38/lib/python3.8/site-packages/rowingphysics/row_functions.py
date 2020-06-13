#from math import *
from __future__ import absolute_import
from __future__ import print_function
import time
import pylab
import numpy as np
from .srnumerical import *
#import scipy
import matplotlib
from matplotlib import pyplot

from numpy import linspace

# Global constants

# water density (kg/m^3)
rho = 1000.

# Maximum Lift coefficient for blade
CLmax = 1.0

def drag_eq(displacement,velo,alfaref=3.5,doprint=0,constantdrag=0):
   corr = alfaref/3.5

   # usual coefficient - boat is a spheroid
   c1 = 1./3.
   c2 = 2./3.
   d1 = 0.06
   d2 = 28.
   d3 = 0.10891
   a1 = 0
   a3 = 0

   beam = a1+d1*displacement**(c1)
   boatlength = d2*beam
   wetted_area = a3+d3*displacement**(c2)

   kinvis = 1.19e-6
   rho = 999.97

   D = (displacement)/rho
   Re = boatlength*velo/kinvis 
   Cf = 0.075/((np.log10(Re)-2)**2)    
   alpha = 0.5*rho*wetted_area*Cf    
   alpha = alpha*corr  # correction, should be 1.0 normally
   a1 = alpha/0.8

   if (constantdrag==1):
      a1 = alfaref

   if doprint==1:
      print("----- Drag resistance data --------")
      print(("Corr : ",corr))
      print(("Beam : ",beam))
      print(("Boat length : ",boatlength))
      print(("Wetted Area : ",wetted_area))
      print(("alpha skin  : ",alpha))
      print(("alpha total : ",a1))
      print("----- Drag resistance data --------")
      print("")
   
   W2 = a1*velo**2

   return W2


def vboat(mc,mb,vc):
   vb = mc*vc/(mc+mb)
   return vb

def vhandle(v,lin,lout,mc,mb):
   gamma = mc/(mc+mb)
   vc = lin*v/(lout+gamma*lin)
   return vc

def d_recovery(dt,v,vc,dvc,mc,mb,alef):

   dv = 0.0

   vb = vboat(mc,mb,vc)
   dvb = vboat(mc,mb,dvc)

   dv = dt*(-alef*(v-vb)**2)/(mb+mc)
   
   return dv

def d_stroke(dt,v,vc,dvc,mc,mb,alef,F):
   dv = 0.0
   vb = vboat(mc,mb,vc)
   dvb = vboat(mc,mb,dvc)
   Ftot = F - alef*(v-vb)**2
   dv = dt*Ftot/(mb+mc)
   
   return dv

def de_footboard(mc,mb,vs1,vs2):
   de = 0.0
   vt = 0.0
   
   vb1 = vboat(mc,mb,vs1)
   vb2 = vboat(mc,mb,vs2)

   vmb1 = vt-vb1   
   vmb2 = vt-vb2

   vmc1 = vt+vs1-vb1   
   vmc2 = vt+vs2-vb2
   
   e_1 = 0.5*(mb*vmb1**2)+0.5*mc*vmc1**2 
   e_2 = 0.5*(mb*vmb2**2)+0.5*mc*vmc2**2 
   e_t = 0.5*(mc+mb)*vt**2 


   de2 = e_2 - e_t 
   de1 = e_1 - e_t 

   if (sign(vs1) == sign(vs2)):
     de = max([e_2 - e_1,0]) 
   else:
     de = max([e_2 - e_t,0]) 

   return de

def blade_force(oarangle,rigging,vb,fblade,doplot=0):

   lin = rigging.lin
   lscull = rigging.lscull
   lout = lscull - lin

   phidot0 = vb*np.cos(oarangle)/lout
   phidot = linspace(phidot0,2*abs(phidot0))


   vblade = phidot*lout
   u1 = vblade-vb*np.cos(oarangle)
   up = vb*np.sin(oarangle)
   u = (u1**2 + up**2)**(0.5)  # fluid velocity
   a = np.arctan(u1/up)   # angle of attack
  
   C_D = 2*CLmax*np.sin(a)**2.
   C_L = CLmax*np.sin(2.*a)
   area = rigging.bladearea
#   area = rigging.bladearea*np.sin(a)
   F_L = 0.5*C_L*rho*area*u**2.0
   F_D = 0.5*C_D*rho*area*u**2.0


   F_R = (F_L**2 + F_D**2)**(0.5)
   F_prop = F_R * np.cos(oarangle)

   phidot1 = sr_interpol1(phidot,F_R,fblade)

   if (doplot==1):
      try:
         pyplot.clf()
         ax1 = pyplot.subplot(111)

         pyplot.plot(phidot,F_R,label='Fn')
         pyplot.plot(phidot,F_prop,label='F_l')
         pyplot.plot(phidot,fblade+0.0*phidot,label='F_target')
         pylab.legend(loc='best')
         pyplot.xlabel("angular velocity (rad/s)")
         pyplot.ylabel('F (N)')

         pyplot.show()
      except NameError:
         print("No plotting today")
  
   vblade = phidot1*lout

   u1 = vblade-vb*np.cos(oarangle)
   up = vb*np.sin(oarangle)

   u = (u1**2 + up**2)**(0.5)  # fluid velocity
   a = np.arctan(u1/up)   # angle of attack

   C_D = 2*CLmax*np.sin(a)**2.
   C_L = CLmax*np.sin(2.*a)
   area = rigging.bladearea
#   area = rigging.bladearea*np.sin(a)
   F_L = 0.5*C_L*rho*area*u**2.0
#   F_D = abs(0.5*C_D*rho*area*u**2.0)
   F_D = 0.5*C_D*rho*area*u**2.0

   F_R = (F_L**2 + F_D**2)**(0.5)
   F_prop = F_R * np.cos(oarangle)

   return [phidot1,F_R,F_prop,F_L,F_D,C_L,C_D,a]

