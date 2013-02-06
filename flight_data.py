import sys
import time
from datetime import datetime
import numpy as np

class FlightData():
    def __init__(self, filename):
        self.initVariables()
        self.loadData(filename)
        
    def initVariables(self):
        
        self.dataBat=[]
        self.dataVX=[]
        self.dataVY=[]
        self.dataVZ=[]
        self.dataYaw=[]
        self.dataPitch=[]
        self.dataRoll=[]
        self.dataAltitude=[]
        self.dataX=[]
        self.dataY=[]
        self.dataZ=[]
        self.dataCommand=[]
        self.dataw1=[]
        self.dataw2=[]
        self.dataw3=[]
        self.dataw4=[]
        self.dataTime=[]
        
        #time, altitude, yaw, pitch, roll, w1, w2, w3, w4, x, y, z
        #self.data = np.array(())
        
            
    def loadData_original(self, filename):
        self.initVariables()
        try:
            f = open(filename, 'r')
            for line in f:
                if line[0]=='[' and not 'None' in line:
                    pos=line.find(', ')
                    self.dataBat.append(line[1:pos])
                    pos2=line.find(', ',pos+1)
                    self.dataVX.append(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataVY.append(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataVZ.append(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataYaw.append(float(line[pos+2:pos2]))
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataPitch.append(float(line[pos+2:pos2]))
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataRoll.append(float(line[pos+2:pos2]))
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataAltitude.append(float(line[pos+2:pos2]))
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataX.append(float(line[pos+2:pos2]))
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataY.append(float(line[pos+2:pos2]))
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    self.dataZ.append(float(line[pos+2:pos2]))
                    pos=line.find("'",pos)
                    pos2=line.find("'",pos+1)
                    command=str(line[pos+1:pos2])
                    self.dataCommand.append(command)
                    if command.find('winkel')<>-1: 
                        p1=command.find('=')+1
                        p2=command.find(',',p1+1)
                        self.dataw1.append(float( command[p1:p2] ) )
                        p1=p2+1
                        p2=command.find(',',p1+1)
                        self.dataw2.append(float( command[p1:p2] ) )
                        p1=p2+1
                        p2=command.find(',',p1+1)
                        self.dataw3.append(float( command[p1:p2] ) )
                        p1=p2+1
                        self.dataw4.append(float( command[p1:] ) )
                    else:
                        self.dataw1.append(0.0)
                        self.dataw2.append(0.0)
                        self.dataw3.append(0.0)
                        self.dataw4.append(0.0)
                    
                    pos=pos2
                    pos2=line.find(']',pos+1)
                    self.dataTime.append(line[pos+3:pos2])

        except:
            print 'Error while loading data'
        print 'Finished loading data. '
        
    def loadData(self, filename):
        self.loadData_original(filename)
        dataTimeDiffs = []
        
        # Zeit in ms
        """
        base_datetime = None
        for timeString in self.dataTime:
            the_time = datetime.fromtimestamp(float(timeString))
            milliseconds_diff = 0
            if base_datetime == None:
                base_datetime = the_time
            else:
                timediff = the_time - base_datetime
                milliseconds_diff = ((timediff.days * 24 * 60 * 60 + timediff.seconds) * 1000 + timediff.microseconds / 1000)
            dataTimeDiffs.append(milliseconds_diff)
        """
        #Zeitdifferenz in ms
        last_time = the_time = datetime.fromtimestamp(float(self.dataTime[0])-0.1)
        for timeString in self.dataTime:
            the_time = datetime.fromtimestamp(float(timeString))
            timediff = the_time - last_time
            milliseconds_diff = ((timediff.days * 24 * 60 * 60 + timediff.seconds) * 1000 + timediff.microseconds / 1000)
            dataTimeDiffs.append(milliseconds_diff)
            last_time = the_time
                        
        self.dataTime = dataTimeDiffs
            
        nr_rows = len(self.dataTime)
        self.data = np.array((nr_rows, 12))
        
        #time, altitude, yaw, pitch, roll, w1, w2, w3, w4, x, y, z
        self.data = np.column_stack((np.asarray(self.dataTime),np.asarray(self.dataAltitude), np.asarray(self.dataYaw),
                                     np.asarray(self.dataPitch), np.asarray(self.dataRoll), np.asarray(self.dataw1),
                                     np.asarray(self.dataw2), np.asarray(self.dataw3), np.asarray(self.dataw4),
                                     np.asarray(self.dataX), np.asarray(self.dataY), np.asarray(self.dataZ)))
        #self.data = np.hstack((self.data, self.dataAltitude, 1))
        #self.data = np.hstack((self.data, self.dataYaw, 2))
        
        #Normalisierung - macht alles noch schlimmer!
        """
        means = self.data.mean(1)
        stds = self.data.std(1)
        self.data = self.data - means[:,None]
        self.data = self.data / stds[:,None]
        print self.data.shape
        #print self.data.mean(1)
        #print self.data.std(1)
        """
        
if __name__ == '__main__':
    flight_data = FlightData('flight_data/a_to_b_constantYaw/flight_Sun_03_Feb_2013_12_27_26_AllData')
    
    #print len(data.dataZ), len(data.dataCommand), len(data.dataw1), len(data.dataPitch)
    #print len(flight_data.dataTime)
