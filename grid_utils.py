import csv
import os
import re
    
def merge_files(result_file_name):
    #file_names = filter(os.path.isfile, os.listdir( os.curdir ) )
    #file_names = filter(os.path.isfile, os.listdir('./tmp' ) )
    prefix = 'tmp/'
    file_names = os.listdir('./'+prefix )
    name_regex = re.compile(r"python\.o.*")
    result_files = filter(name_regex.search, file_names)
    print len(result_files), 'files to merge found'
    
    if len(result_files) == 0:
        return
    #read values as dictionaries
    rows = []
    for file_name in result_files:
        with open(prefix +file_name, 'rb') as f:
            #reader = csv.reader(f)
            reader = csv.DictReader(f)
            #csv_file = csv.DictReader(open(test_file, 'rb'), delimiter=',', quotechar='"')
            for row in reader:
                rows.append(row)
    #sort
    rows = sorted(rows, key=lambda k: float(k['NRMSE']))             
    
    #write values into a single file
    fieldnames = rows[0].keys()
    if 'NRMSE' in fieldnames:
        fieldnames.remove('NRMSE')
        fieldnames.append('NRMSE')
    #print 'Writing result to ', result_file_name
    results_file = open('results/'+result_file_name,'wb')
    writer = csv.DictWriter(results_file, fieldnames)
    writer.writerow(dict((fn,fn) for fn in fieldnames))
    writer.writerows(rows)
    results_file.close()
    
    
    #check if the file was written
    file_names = os.listdir( 'results' )
    if result_file_name in file_names:
        for partial_result in result_files:
            os.remove(prefix +partial_result)
            
if __name__ == '__main__':
    result_file_name = "results.csv"
    merge_files("results.csv") 
