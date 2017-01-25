from nose.tools import assert_equals
from nose import with_setup
import rowingdata
import datetime
import numpy as np
import pandas as pd
from nose_parameterized import parameterized
import unittest

class TestBasicRowingData:
    row = rowingdata.rowingdata(csvfile='testdata/testdata.csv')

    def test_filetype(self):
        assert_equals(rowingdata.get_file_type('testdata/testdata.csv'),'csv')
    
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
    def testergdata(self):
        csvfile = 'testdata/ergdata_example.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'ergdata')
        r = rowingdata.ErgDataParser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,180)
        totaldist = row.df['cum_dist'].max()
        assert_equals(totaldist,1992)
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(totaltime,520)

class TestpainsledDesktopParser:
    def testpainsleddesktop(self):
        csvfile = 'testdata/painsled_desktop_example.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'painsleddesktop')
        r = rowingdata.painsledDesktopParser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,638)
        totaldist = row.df['cum_dist'].max()
        assert_equals(totaldist,7097)
        assert_equals(row.rowdatetime,datetime.datetime(2016,3,29,16,41,27))
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(totaltime,1802)

class TestBoatCoachParser:
    def testboatcoach(self):
        csvfile = 'testdata/boatcoach.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'boatcoach')
        r = rowingdata.BoatCoachParser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,119)
        totaldist = row.df['cum_dist'].max()
        assert_equals(totaldist,499)
        assert_equals(row.rowdatetime,datetime.datetime(2016,11,28,7,37,2))
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(totaltime,118)
    
class TestspeedcoachParser:
    def testspeedcoach(self):
        csvfile = 'testdata/speedcoachexample.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'speedcoach')
        r = rowingdata.speedcoachParser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,476)
        totaldist = row.df['cum_dist'].max()
        assert_equals(totaldist,9520)
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(totaltime,3176.5)
    
class TestErgStickParser:
    def testergstick(self):
        csvfile = 'testdata/ergstick.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'ergstick')
        r = rowingdata.ErgStickParser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,2400)
        totaldist = row.df['cum_dist'].max()
        assert_equals(int(totaldist),4959)
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(int(totaltime),1201)
    
class TestMysteryParser:
    def testmystery(self):
        csvfile = 'testdata/mystery.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'mystery')
        r = rowingdata.MysteryParser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,4550)
        totaldist = row.df['cum_dist'].max()
        assert_equals(totaldist,7478)
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(int(totaltime),2325)
    
class TestRowProParser:
    def testrowpro(self):
        csvfile = 'testdata/RP_testdata.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'rp')
        r = rowingdata.RowProParser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,988)
        totaldist = row.df['cum_dist'].max()
        assert_equals(totaldist,10000)
        assert_equals(row.rowdatetime,datetime.datetime(2016,3,15,18,49,48))
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(int(10*totaltime),22660)

class TestRowProParserIntervals:
    def testrowprointervals(self):
        csvfile = 'testdata/RP_interval.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'rp')
        r = rowingdata.RowProParser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,1674)
        totaldist = row.df['cum_dist'].max()
        assert_equals(int(totaldist),19026)
        assert_equals(row.rowdatetime,datetime.datetime(2016,1,12,18,23,10))
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(totaltime,4800)
    
class TestSpeedCoach2Parser:
    def testspeedcoach2(self):
        csvfile = 'testdata/SpeedCoach2example.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'speedcoach2')
        r = rowingdata.SpeedCoach2Parser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,97)
        totaldist = row.df['cum_dist'].max()
        assert_equals(int(10*totaldist),7516)
        assert_equals(row.rowdatetime,datetime.datetime(2016,7,28,11,35,1,500000))
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(totaltime,170)
    
class TestSpeedCoach2_v127Parser:
    def testspeedcoach2v127(self):
        csvfile = 'testdata/SpeedCoach2Linkv1.27.csv'
        assert_equals(rowingdata.get_file_type(csvfile),'speedcoach2')
        r = rowingdata.SpeedCoach2Parser(csvfile=csvfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,1408)
        totaldist = row.df['cum_dist'].max()
        assert_equals(totaldist,14344.5)
        assert_equals(row.rowdatetime,datetime.datetime(2016,11,5,10,2,3,200000))
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(int(10*totaltime),45018)
    
class TestFITParser:
    def testfit(self):
        fitfile = 'testdata/3x250m.fit'
        assert_equals(rowingdata.get_file_type(fitfile),'fit')
        r = rowingdata.FITParser(fitfile)
        row = rowingdata.rowingdata(df=r.df)
        assert_equals(row.number_of_rows,94)
        totaldist = row.df['cum_dist'].max()
        assert_equals(int(totaldist),750)
        assert_equals(row.rowdatetime,datetime.datetime(2016,7,28,9,35,29))
        totaltime = row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
        assert_equals(int(10*totaltime),4870)

class TestSequence(unittest.TestCase):
    list = pd.read_csv('testdata/testdatasummary.csv')
    lijst = []
    for i in list.index:
        filename = list.ix[i,'filename']
        expected = list.ix[i,1:]
        lijst.append(
            (filename,filename,expected)
            )

    @parameterized.expand(lijst)
    
    def test_check(self, name, filename, expected):
        f2 = 'testdata/'+filename
        res = rowingdata.checkdatafiles.checkfile(f2)
        if res != 0:
            for key,value in res.iteritems():
                if expected[key] != 0:
                    assert_equals(value,expected[key])        


                      
