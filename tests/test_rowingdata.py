from nose.tools import assert_equals
import rowingdata
import datetime

def test_basic_rowingdata():
    row = rowingdata.rowingdata('../testdata/testdata.csv')
    assert_equals(row.rowtype,'Indoor Rower')
    assert_equals(row.dragfactor,104.42931937172774)
    assert_equals(row.number_of_rows,191)
    assert_equals(row.rowdatetime,datetime.datetime(2016,5,20,13,41,26,962390))

                  
