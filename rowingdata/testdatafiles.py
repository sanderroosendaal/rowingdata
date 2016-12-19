import rowingdata
from rowingdata import TCXParser,RowProParser,ErgDataParser,TCXParserNoHR
from rowingdata import BoatCoachParser
from rowingdata import MysteryParser
from rowingdata import painsledDesktopParser,speedcoachParser,ErgStickParser
from rowingdata import SpeedCoach2Parser,FITParser,fitsummarydata

    

def testfile(f2):
    fileformat = rowingdata.get_file_type(f2)
    print fileformat

    if len(fileformat)==3 and fileformat[0]=='zip':
        with zipfile.ZipFile(f2) as z:
            f = z.extract(z.namelist()[0],path='C:/Downloads')
            fileformat = fileformat[2]
            os.remove(f_to_be_deleted)
            
    if fileformat == 'unknown':
	return 0

    # handle non-Painsled
    if (fileformat != 'csv'):
	# handle RowPro:
	if (fileformat == 'rp'):
	    row = RowProParser(f2)
	    # handle TCX
	if (fileformat ==  'tcx'):
	    row = TCXParser(f2)

	# handle Mystery
	if (fileformat == 'mystery'):
	    row = MysteryParser(f2)

	# handle TCX no HR
	if (fileformat == 'tcxnohr'):
	    row = TCXParserNoHR(f2)
		    
	# handle ErgData
	if (fileformat == 'ergdata'):
	    row = ErgDataParser(f2)

	# handle BoatCoach
	if (fileformat == 'boatcoach'):
	    row = BoatCoachParser(f2)

	# handle painsled desktop
	if (fileformat == 'painsleddesktop'):
	    row = painsledDesktopParser(f2)

	# handle speed coach GPS
	if (fileformat == 'speedcoach'):
	    row = speedcoachParser(f2)

	# handle speed coach GPS 2 
	if (fileformat == 'speedcoach2'):
	    row = SpeedCoach2Parser(f2)
    
	# handle ErgStick
        if (fileformat == 'ergstick'):
	    row = ErgStickParser(f2)
		    
	# handle FIT
	if (fileformat == 'fit'):
	    row = FITParser(f2)


        f_out = 'C:/Downloads/'+f2[12:-4]+'o.csv'
        row.write_csv(f_out,gzip=True)
        f2 = f_out

    row = rowingdata.rowingdata(f2)

    print "nr lines",row.number_of_rows
    print "data ",row.rowdatetime
    print "dist ",row.df['cum_dist'].max()
    print "Time ",row.df['TimeStamp (sec)'].max()-row.df['TimeStamp (sec)'].min()
    print "Nr intervals ",len(row.df[' lapIdx'].unique())

    res =  row.intervalstats_values()
    times = res[0]
    distance = res[1]


    print "Interval 1 time ",times[0]
    print "Interval 1 dist ",distance[0]

    

        
