from __future__ import absolute_import
from math import *

class rigging:
   def __init__(self,lin=0.9,mb=14,lscull=2.885,span=1.60,spread=0.88,roworscull='scull',
                catchangle=-0.93,bladearea=822.e-4,bladelength = 0.46,
                Nrowers=1,dragform=1.0):  # catch angle in radian
      self.lin = lin
      self.mb = mb
      self.bladelength = bladelength
      self.lscull = lscull-0.5*bladelength
      self.Nrowers = Nrowers
      self.roworscull = roworscull
      self.span= span
      self.__spread = spread
      self.__bladearea = bladearea
      self.catchangle = catchangle
      self.dragform = dragform

   @property
   def spread(self):
      if (self.roworscull == 'scull'):
         return self.span/2.
      else:
         return self.__spread

   @property
   def overlap(self):
      if (self.roworscull == 'scull'):
         return 2.*self.lin-self.span
      else:
         return self.lin-self.__spread

   @property
   def buitenhand(self):
      if (self.roworscull == 'scull'):
         return self.span-2.*self.lin*cos(self.catchangle)
      else:
         return self.spread-self.lin*cos(self.catchangle)

   @property
   def bladearea(self):
      if (self.roworscull == 'scull'):
         return self.__bladearea*2
      else:
         return self.__bladearea

   @property
   def dcatch(self): 
      return self.lin*sin(self.catchangle)
      
   def oarangle(self,x):  
      dist = self.dcatch+x
      try:
          angle = asin(dist/self.lin)
      except ValueError:
          angle = pi/2
          
      return angle

      
