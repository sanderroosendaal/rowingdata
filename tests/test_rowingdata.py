from nose.tools import assert_equals
from nose import with_setup
import rowingdata
import datetime
import numpy as np

class TestBasicRowingData:
    row = rowingdata.rowingdata(csvfile='testdata/testdata.csv')

    def test_basic_rowingdata(self):
        assert_equals(self.row.rowtype,'Indoor Rower')
        assert_equals(self.row.dragfactor,104.42931937172774)
        assert_equals(self.row.number_of_rows,191)
        assert_equals(self.row.rowdatetime,datetime.datetime(2016,5,20,13,41,26,962390))
        totaldist = self.row.df['cum_dist'].max()
        totaltime = self.row.df['TimeStamp (sec)'].max()-self.row.df['TimeStamp (sec)'].min()
        totaltime = totaltime+self.row.df.ix[0,' ElapsedTime (sec)']
        assert_equals(totaltime, 540.04236011505122)
        assert_equals(totaldist, 2000)

    def test_intervals_rowingdata(self):
        ts,ds,st = self.row.intervalstats_values()
        assert_equals(ts,[139.7,0.0,130.1,0.0,134.4,0.0,132.8,0.0])
        assert_equals(ds,[510,0,499,0,498,0,491,0])
        assert_equals(st,[4,3,4,3,4,3,4,3])
        sum = int(10*np.array(ts).sum())/10.
        assert_equals(sum,537.0)
        self.row.updateinterval_string('4x500m')
        assert_equals(len(self.row.df[' lapIdx'].unique()),4)
        ts,ds,st = self.row.intervalstats_values()
        assert_equals(ts,[137.0,0.0,130.2,0.0,134.8,0.0,135.1,0.0])
        assert_equals(ds,[500,0,500,0,500,0,500,0])
        assert_equals(st,[5,3,5,3,5,3,5,3])
        sum = int(10*np.array(ts).sum())/10.
        assert_equals(sum,537.0)
                      
        
class TestErgData:
    r = rowingdata.ErgDataParser(csvfile='testdata/ergdata_example.csv')
    row = rowingdata.rowingdata(df=r.df)
    assert_equals(row.number_of_rows,180)
    totaldist = row.df['cum_dist'].max()
    assert_equals(totaldist,1992)
    totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
    assert_equals(totaltime,520)

class TestpainsledDesktopParser:
    r = rowingdata.painsledDesktopParser(csvfile='testdata/painsled_desktop_example.csv')
    row = rowingdata.rowingdata(df=r.df)
    assert_equals(row.number_of_rows,638)
    totaldist = row.df['cum_dist'].max()
    assert_equals(totaldist,7097)
    assert_equals(row.rowdatetime,datetime.datetime(2016,3,29,16,41,27))
    totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
    assert_equals(totaltime,1802)

class TestBoatCoachParser:
    r = rowingdata.BoatCoachParser(csvfile='testdata/boatcoach.csv')
    row = rowingdata.rowingdata(df=r.df)
    assert_equals(row.number_of_rows,132)
    totaldist = row.df['cum_dist'].max()
    assert_equals(totaldist,499)
    assert_equals(row.rowdatetime,datetime.datetime(2016,11,28,7,37,2))
    totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
    assert_equals(totaltime,118)
    
