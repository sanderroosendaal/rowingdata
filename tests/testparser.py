from __future__ import absolute_import
from rowingdata.trainingparser import parse
from nose.tools import assert_equal
import unittest

class TestParser:
    def test8x500m(self):
        res=parse("8x500m/2min")
        assert_equal(len(res),48)
        assert_equal(res[0],500)
        assert_equal(res[1],'meters')
        assert_equal(res[2],'work')
        assert_equal(res[3],120)
        assert_equal(res[4],'seconds')
        assert_equal(res[5],'rest')

    def test10km(self):
        res=parse("10km")
        assert_equal(len(res),3)
        assert_equal(res[0],1e4)
        assert_equal(res[1],'meters')
        assert_equal(res[2],'work')

    def testsixmins(self):
        res=parse("6min/4min+5min/3min+3min/3min+3min")
        assert_equal(len(res),21)
        assert_equal(res[0],360)
        assert_equal(res[1],'seconds')
        assert_equal(res[2],'work')
        assert_equal(res[3],240)
        assert_equal(res[4],'seconds')
        assert_equal(res[5],'rest')

    def testfourtimes1km(self):
        res=parse("4x((500m+500m)/5min)")
        assert_equal(len(res),36)
        assert_equal(res[0],500)
        assert_equal(res[1],'meters')
        assert_equal(res[2],'work')
        assert_equal(res[3],500)
        assert_equal(res[4],'meters')
        assert_equal(res[5],'work')
        assert_equal(res[6],300)
        assert_equal(res[7],'seconds')
        assert_equal(res[8],'rest')

    def testtwokonoff(self):
        res=parse("2x500m/500m")
        assert_equal(len(res),12)
        assert_equal(res[0],500)
        assert_equal(res[1],'meters')
        assert_equal(res[2],'work')
        assert_equal(res[3],500)
        assert_equal(res[4],'meters')
        assert_equal(res[5],'rest')

    def testgreg1(self):
        res=parse("3x(1000m+750m+500m+250m)+2500m")
        assert_equal(len(res),39)
        assert_equal(res[0],1000)
        assert_equal(res[1],'meters')
        assert_equal(res[2],'work')
        assert_equal(res[3],750)
        assert_equal(res[4],'meters')
        assert_equal(res[5],'work')

    def testgreg2(self):
        res=parse("3x((1000m+750m+500m+250m)/0m)+2500m")
        assert_equal(len(res),48)
        assert_equal(res[0],1000)
        assert_equal(res[1],'meters')
        assert_equal(res[2],'work')
        assert_equal(res[3],750)
        assert_equal(res[4],'meters')
        assert_equal(res[5],'work')

    def testgreg3(self):
        res = parse("1min + 9 x (10sec + 50 sec)")
        assert_equal(len(res),57)
        assert_equal(res[0],60)
        assert_equal(res[3],10)
        assert_equal(res[6],50)
        assert_equal(res[9],10)
        assert_equal(res[12],50)

    def testgreg4(self):
        res = parse("1min + 10sec + 50 sec + 8x1min")
        assert_equal(len(res),33)
        assert_equal(res[0],60)
        assert_equal(res[3],10)
        assert_equal(res[6],50)
        assert_equal(res[9],60)


    def testgreg5(self):
        res = parse("1min + 1x(10sec+50sec) + 8 x 1min")
        assert_equal(len(res),33)
        assert_equal(res[0],60)
        assert_equal(res[3],10)
        assert_equal(res[6],50)
        assert_equal(res[9],60)

    def testgreg6(self):
        res = parse("1min + 2x(10sec+50sec) + 7x1min")
        assert_equal(len(res),36)
        assert_equal(res[0],60)
        assert_equal(res[3],10)
        assert_equal(res[6],50)
        assert_equal(res[9],10)
        assert_equal(res[12],50)
