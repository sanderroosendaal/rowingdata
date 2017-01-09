from rowingdata.trainingparser import parse
from nose.tools import assert_equals
import unittest

class TestParser:
    def test8x500m(self):
        res = parse("8x500m/2min")
        assert_equals(len(res),48)
        assert_equals(res[0],500)
        assert_equals(res[1],'meters')
        assert_equals(res[2],'work')
        assert_equals(res[3],120)
        assert_equals(res[4],'seconds')
        assert_equals(res[5],'rest')

    def test10km(self):
        res = parse("10km")
        assert_equals(len(res),3)
        assert_equals(res[0],1e4)
        assert_equals(res[1],'meters')
        assert_equals(res[2],'work')

    def testsixmins(self):
        res = parse("6min/4min+5min/3min+3min/3min+3min")
        assert_equals(len(res),21)
        assert_equals(res[0],360)
        assert_equals(res[1],'seconds')
        assert_equals(res[2],'work')
        assert_equals(res[3],240)
        assert_equals(res[4],'seconds')
        assert_equals(res[5],'rest')

    def testfourtimes1km(self):
        res = parse("4x(500m+500m)/5min")
        assert_equals(len(res),36)
        assert_equals(res[0],500)
        assert_equals(res[1],'meters')
        assert_equals(res[2],'work')
        assert_equals(res[3],500)
        assert_equals(res[4],'meters')
        assert_equals(res[5],'work')
        assert_equals(res[6],300)
        assert_equals(res[7],'seconds')
        assert_equals(res[8],'rest')

    def testtwokonoff(self):
        res = parse("2x500m/500m")
        assert_equals(len(res),12)
        assert_equals(res[0],500)
        assert_equals(res[1],'meters')
        assert_equals(res[2],'work')
        assert_equals(res[3],500)
        assert_equals(res[4],'meters')
        assert_equals(res[5],'rest')

    def testgreg1(self):
        res = parse("3x(1000m+750m+500m+250m)+2500m")
        assert_equals(len(res),39)
        assert_equals(res[0],1000)
        assert_equals(res[1],'meters')
        assert_equals(res[2],'work')
        assert_equals(res[3],750)
        assert_equals(res[4],'meters')
        assert_equals(res[5],'work')

    def testgreg2(self):
        res = parse("3x(1000m+750m+500m+250m)/0m+2500m")
        assert_equals(len(res),48)
        assert_equals(res[0],1000)
        assert_equals(res[1],'meters')
        assert_equals(res[2],'work')
        assert_equals(res[3],750)
        assert_equals(res[4],'meters')
        assert_equals(res[5],'work')
        
