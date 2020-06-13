from __future__ import absolute_import
from math import *
from numpy import *
from six.moves import range

# force functions
class strongmiddle:
    def __init__(self,frac=0.5):
        self.frac = frac

    def forceprofile(self,favg,x):
        f = favg+0 
        f = (self.frac*favg*pi*sin(pi*x)/2.)+(1.-self.frac)*favg
        return f

    def maxforce(self,favg):
        return favg*(self.frac*pi/2.+(1.-self.frac))

class strongmiddle2:
   def __init__(self,frac=0.5):
      self.frac = frac

   def forceprofile(self,favg,x):
      f = favg+0 
      f = (self.frac*favg*pi*sin(pi*x)/2.)+2*(1.-self.frac)*favg*(1-x)
      return f

class flat:
   def forceprofile(self,favg,x):
      f = favg
      return f

class trapezium:
   def __init__(self,h1 = 1.0, h2 = 1.0, x1 = 0.25, x2 = 0.75):
      self.ratio = h1*0.5*x2 + h2*(0.5 - 0.5*x1)
      self.h1 = h1
      self.h2 = h2
      self.x1 = x1
      self.x2 = x2

   def forceprofile(self,favg,x):
      f = 0
      if (x<self.x1):
         f = favg*self.h1*x/self.x1
      elif (x>self.x2):
         f = favg*self.h2*(1.-x)/(1.-self.x2)
      else:
         f = (self.h1 + (self.h2-self.h1)*(x-self.x1)/(self.x2-self.x1))*favg

      f = f/self.ratio
      return f


class trapezium2:
   def __init__(self,h1 = 1.0, h2 = 1.0, x1 = 0.25, x2 = 0.75, h0 = 0.5):
      self.ratio = h1*0.5*x2 + h2*(0.5 - 0.5*x1) + h0
      self.ratio2 = h1*0.5*x2 + h2*(0.5 - 0.5*x1)
      self.frac = self.ratio2/self.ratio
      self.h1 = h1
      self.h2 = h2
      self.x1 = x1
      self.h0 = h0
      self.x2 = x2

   def forceprofile(self,favg,x):
      f = 0
      if (x<self.x1):
         f = favg*self.h1*x/self.x1
      elif (x>self.x2):
         f = favg*self.h2*(1.-x)/(1.-self.x2)
      else:
         f = (self.h1 + (self.h2-self.h1)*(x-self.x1)/(self.x2-self.x1))*favg

      f = self.frac*f+(1-self.frac)*favg*self.h0
      f = f/self.ratio
      return f


class fromfile:
    def __init__(self,fname = 'empforce.txt'):
        self.fname = fname
        data = genfromtxt(self.fname, delimiter = ',')
        
        self.x = data[:,0]
        self.hpos = self.x/max(self.x)
        self.f = data[:,1]
        self.force = self.f/mean(self.f)
        
        
    def forceprofile(self,favg,x):
        f = 0
        if (x!=0):
            wh2 = max(where(self.hpos<x)[0])
            f = self.force[wh2]*favg
            
        return f

class strongbegin:
   def __init__(self,frac=0.5):
      self.frac = frac

   def forceprofile(self,favg,x):
      f = favg+0 
      f = (2*self.frac*(1.0-x)+(1.-self.frac))*favg
      return f

class strongend:
   def __init__(self,frac=0.5):
      self.frac = frac

   def forceprofile(self,favg,x):
      f = favg+0 
      f = (2*self.frac*x+(1.-self.frac))*favg 
      return f

# recovery functions
class flatrecovery:
   def vhandle(self,vavg,trecovery,time):
      f = -vavg 
      return f
   
   def dxhandle(self,vavg,trecovery,time):
      dx = -time/trecovery
      return dx


class sinusrecovery:
   def vhandle(self,vavg,trecovery,time):
      vhandmax = -pi*vavg/2.
      vhand = vhandmax*sin(pi*time/trecovery)
      return vhand

   def dxhandle(self,vavg,trecovery,time):
      vhandmax = -pi*vavg/2.
      dx = 0.5*(cos(pi*time/trecovery)-1)
      return dx

class sinusrecovery2:
    def __init__(self,strokelength,p1=1.0):
        self.p1 = p1
        self.w1 = pi/(p1)
        self.strokelength = strokelength
        
    def vhandle(self,vavg,trecovery,time):
        w = self.w1/trecovery
        vhandmax = w*self.strokelength/(1-cos(w*trecovery))
        vhand = -vhandmax*sin(w*time)
        return vhand

    def dxhandle(self,vavg,trecovery,time):
        w = self.w1/trecovery
        vhandmax = w*self.strokelength/(1-cos(w*trecovery))
        dx = vhandmax*(cos(w*time)-1)/(w*self.strokelength)
        return dx

class cosinusrecovery:
    def __init__(self,strokelength, p1 = 0.95):
        self.p1 = p1
        self.w1 = pi/(2.0*p1)
        self.strokelength = strokelength
        
    def vhandle(self,vavg,trecovery,time):
        w = self.w1/trecovery
        vhmax = w*self.strokelength/(sin(w*trecovery))
        vhand = -vhmax*cos(w*time)

        return vhand

    def dxhandle(self,vavg,trecovery,time):
        w = self.w1/trecovery
        vhmax = w*self.strokelength/(sin(w*trecovery))
        dxhandle = -vhmax*sin(w*time)/(self.strokelength*w)

        return dxhandle

class genericrecovery:
    def __init__(self, strokelength, As=array([1.0]), ws=array([1.0]), phis=array([0.0])):
        self.As = As
        self.ws = ws
        self.phis = phis
        self.strokelength = strokelength

    def vhandle(self,vavg,trecovery,time):
        v = 0
        dmax = 0

        for index in range(len(self.As)):
            v = v + self.As[index]*exp(1j*self.ws[index]*time+self.phis[index])
            dmax = dmax+(1/1j)*(self.As[index]/self.ws[index])*(exp(1j*self.ws[index]*trecovery+self.phis[index])-exp(1j*self.phis[index]))


        dmaxr = real(dmax)
        vhand = -real(v)*self.strokelength/dmaxr
        return vhand

    def dxhandle(self,vavg,trecovery,time):
        dx = 0
        dmax = 0
        for index in range(len(self.As)):
            dx = dx + (self.As[index]/self.ws[index])*(exp(1j*self.ws[index]*trecovery+self.phis[index])-exp(1j*self.phis[index]))
            dmax = dmax+(1/1j)*(self.As[index]/self.ws[index])*(exp(1j*self.ws[index]*trecovery+self.phis[index])-exp(1j*self.phis[index]))
            
            
        dmaxr = real(dmax)
        dxhandle = -real(dx)*self.strokelength/dmaxr
        return dxhandle

class combirecovery:
    def __init__(self,strokelength, p1 = 0.95, q1 = 0.5):
        self.p1 = p1
        self.q1 = q1
        self.q2 = 1-q1
        self.w1 = pi/(p1)
        self.w2 = pi/(2*p1)
        self.strokelength = strokelength

    def vhandle(self,vavg,trecovery,time):
        w = self.w1/trecovery
        vhandmax = w*self.strokelength/(1-cos(w*trecovery))
        vhand1 = -vhandmax*sin(w*time)
        w = self.w2/trecovery
        vhmax = w*self.strokelength/(sin(w*trecovery))
        vhand2 = -vhmax*cos(w*time)

        return self.q1*vhand1+self.q2*vhand2
        
    def dxhandle(self,vavg,trecovery,time):
        w = self.w1/trecovery
        vhandmax = w*self.strokelength/(1-cos(w*trecovery))
        dx1 = vhandmax*(cos(w*time)-1)/(w*self.strokelength)
        w = self.w2/trecovery
        vhmax = w*self.strokelength/(sin(w*trecovery))
        dx2 = -vhmax*sin(w*time)/(self.strokelength*w)

        return self.q1*dx1+self.q2*dx2

class trianglerecovery:
   def __init__(self,x1 = 0.3):
      self.x1 = x1

   def vhandle(self,vavg,trecovery,time):
      vhand = 0
      trel = time/trecovery
      if (trel<self.x1):
         vhand = -2*vavg*trel/self.x1
      else:
         vhand = -2*vavg*(1.-trel)/(1-self.x1)

      return vhand

   def dxhandle(self,vavg,trecovery,time):
      trel = time/trecovery
      if (trel< self.x1):
         dx = trel**2/self.x1
      else:
         dx = self.x1
         dx -= ((1.-trel)**2)/(1-self.x1)
         dx += (1-self.x1)
         
      return -dx


class realisticrecovery:
   def vhandle(self,vavg,trecovery,time):
      vhandmax = -vavg/0.9
      trel = time/trecovery
      if (trel < 0.1):
         vhand = vhandmax*trel*10.
      elif (trel > 0.9):
         vhand = vhandmax*(1.-trel)*10.
      else:
         vhand = vhandmax

      return vhand
   
   def dxhandle(self,vavg,trecovery,time):
      trel = time/trecovery
      if (trel < 0.1):
         dx = 5*trel**2
      elif (trel > 0.9):
         dx = 5*(0.1**2)
         dx += 0.8
         dx += 5*(0.1**2)-5*(1-trel)**2
      else:
         dx = 5*(0.1**2)
         dx += (trel-0.1)

      return -dx/0.9

# technique functions (CM speed vs handle position)
class technique_meas:
   def vcm(self, vhandle, strokelength, xhandle):
       vc = vhandle
       if (xhandle>=0):
           xr = (xhandle/strokelength) 
           # vc = vhandle*(1-xr)
           vc = 0.85*vhandle-0.75*vhandle*xr**2
           
       return vc

   def vcma(self, vhandle, strokelength, xhandle):
       vc = 0*xhandle
       xr = 0*xhandle
       wh = where(xhandle<0)[0]
       xhandle[wh] = 0*xhandle[wh]
   
       xr = (xhandle/strokelength) 
       # vc = vhandle*(1-xr)
       vc = 0.85*vhandle-0.75*vhandle*xr**2
       vc[wh] = vhandle[wh]
           
       return vc

   def vha(self,vcm, strokelength, xhandle):
      xr = xhandle/strokelength
      vh = vcm/(0.85-0.75*xr**2)

      return vh

# the real "crew" class
class crew:
   def __init__(self,mc=80.,strokelength=1.4,tempo=30.0,frac=0.5,
                recprofile=sinusrecovery(),
                strokeprofile=trapezium(x1=0.15,x2=0.5,h2=0.9), 
                technique = technique_meas(), 
                maxpower = 1000., maxforce = 1000.):
       
      self.mc = mc
      self.maxforce = maxforce
      self.strokelength=strokelength
      self.tempo = tempo
      self.recprofile = recprofile
      self.strokeprofile = strokeprofile
      self.technique = technique
      self.maxpower = maxpower

   def forceprofile(self,favg,x):
      return self.strokeprofile.forceprofile(favg,x/self.strokelength)

   def vhandle(self,vavg,trecovery,time):
      return self.recprofile.vhandle(vavg,trecovery,time)

   def dxhandle(self,vavg,trecovery,time):
      return self.recprofile.dxhandle(vavg,trecovery,time)

   def vcm(self,vhandle, xhandle):
      return self.technique.vcm(vhandle, self.strokelength, xhandle)

   def vcma(self,vhandle, xhandle):
      return self.technique.vcma(vhandle, self.strokelength, xhandle)

   def vha(self,vcm, xhandle):
      return self.technique.vha(vcm, self.strokelength, xhandle)
