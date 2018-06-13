from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd

x1=np.array([1,3,4])
y1=x1*2
x2=np.array([1,2,3,4,5,6,7])
y2=x2*2-1

def test():
    x=np.concatenate((x1,x2))
    y=np.concatenate((y1,y2))
    print((len(x),len(y)))
    data=pd.DataFrame({'x':x,
                         'y':y})
    data=data.drop_duplicates(subset='x').sort('x',ascending=1)

    return data
