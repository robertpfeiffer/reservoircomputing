import sys
import time
from datetime import datetime
from py_utils import *
import numpy as np
import esn_persistence

class FlightData():
    def __init__(self, filename, load_altitude=False, load_dV=False, load_xyz=True):
        if filename is not None:
            self.load_file(filename, load_altitude, load_dV, load_xyz)
        
    def load_file(self, filename, load_altitude=False, load_dV=False, load_xyz=True):
        self.initVariables()
        self.load_altitude=load_altitude
        self.load_dV = load_dV
        self.load_xyz = load_xyz
        self.loadData(filename)
        
        #18 Dim:
        #time, vx, vy, vz, yaw, pitch, roll, altitude, x, y, z, targetX, targetY, targetZ, w1, w2, w3, w4
        all_dims = 18
        self.xyz_columns = np.arange(8,11)
        self.target_xyz_columns = np.arange(11, 14)
        self.w_columns = np.arange(14, 18)
        
        if not load_dV:
            self.xyz_columns -= 3
            self.target_xyz_columns -= 3
            self.w_columns -= 3
            all_dims-=3
            
        if not load_altitude:
            self.xyz_columns -= 1
            self.target_xyz_columns -= 1
            self.w_columns -= 1
            all_dims -= 1
            
        if not load_xyz:
            self.xyz_columns = []
            self.target_xyz_columns = []
            self.w_columns -= 6
            all_dims -= 6
        else:
            #for prediction
            self.w_non_target_columns = np.hstack((np.arange(0, all_dims-7), self.w_columns))
            
        #self.xyz_columns = np.arange(8,11)
        #self.target_xyz_columns = np.arange(11, 14)
        
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
        self.dataTargetX = []
        self.dataTargetY = []
        self.dataTargetZ = []
        
        self.containsTarget = False
        self.dataTimeDiffs = []
                
    def loadData_original(self, filename):
        self.initVariables()
        lines = esn_persistence.load_file(filename)
        i = 0
        for line in lines:
            try:
                i += 1
                #print i
                #Battery, VX, VY, VZ, Yaw, Pitch, Roll, Altitude, X, Y, Z, command, target, time
                if line[0]=='[' and not 'None' in line and not 'end' in line:
                    pos=line.find(', ')
                    dataBatValue = line[1:pos]
                    pos2=line.find(', ',pos+1)
                    vxValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    vyValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    vzValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    yawValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    pitchValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    rollValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    altitudeValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    xValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    yValue = float(line[pos+2:pos2])
                    pos=pos2
                    pos2=line.find(', ',pos+1)
                    zValue = float(line[pos+2:pos2])
                    pos=line.find("'",pos)
                    pos2=line.find("'",pos+1)
                    command=str(line[pos+1:pos2])
                    commandValue = command[:]
                    target = None
                    if 'target' in command:
                        parts = command.split(';')
                        command = parts[0]
                        target = parts[1]
                   
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
                    
                    if target is not None:
                        self.containsTarget = True
                        p1 = target.find('[')+1
                        p2=target.find(',',p1)
                        self.dataTargetX.append(float(target[p1:p2] ) )
                        p1=p2+1
                        p2=target.find(',',p1+1)
                        self.dataTargetY.append(float( target[p1:p2] ) )
                        p1=p2+1
                        p2=target.find(',',p1+1)
                        self.dataTargetZ.append(float( target[p1:p2] ) )
                        
                    pos=pos2
                    pos2=line.find(']',pos+1)
                    self.dataTime.append(line[pos+3:pos2])
                    
                    self.dataBat.append(dataBatValue)
                    self.dataVX.append(vxValue)
                    self.dataVY.append(vyValue)
                    self.dataVZ.append(vzValue)
                    self.dataYaw.append(yawValue)
                    self.dataPitch.append(pitchValue)
                    self.dataRoll.append(rollValue)
                    self.dataAltitude.append(altitudeValue)
                    self.dataX.append(xValue)
                    self.dataY.append(yValue)
                    self.dataZ.append(zValue)
                    self.dataCommand.append(commandValue)
                
            except:
                print 'Error while loading data: ',sys.exc_info()[0], sys.exc_info()[1]
        print 'Finished loading data. '
        
    def loadData(self, filename):
        self.loadData_original(filename)
        
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
                
        k = 30
        Vscale = 1000
        Tscale = 100.0
        Yaw_scale = 100.0
        
        #Zeitdifferenz in ms
        last_time = the_time = datetime.fromtimestamp(float(self.dataTime[0])-0.1)
        for timeString in self.dataTime:
            milliseconds_diff = float(compute_time_diff_in_ms(last_time, timeString))
            #the_time = datetime.fromtimestamp(float(timeString))
            #timediff = the_time - last_time
            #milliseconds_diff = ((timediff.days * 24 * 60 * 60 + timediff.seconds) * 1000 + timediff.microseconds / 1000)
            self.dataTimeDiffs.append(milliseconds_diff/Tscale)
            last_time = datetime.fromtimestamp(float(timeString))

        nr_rows = len(self.dataTimeDiffs)
        
        #self.dataTimeDiffs = np.ones(nr_rows)
        
        #self.data = np.array((nr_rows, 15))
        
        #Target ist die Position in k Schritten
        self.dataTargetX = np.asarray(self.dataX)[k:]
        self.dataTargetY = np.asarray(self.dataY)[k:]
        self.dataTargetZ = np.asarray(self.dataZ)[k:]
        #self.dataTargetX = np.asarray(self.dataX)[k:] - np.asarray(self.dataX)[:-k]
        #self.dataTargetY = np.asarray(self.dataY)[k:] - np.asarray(self.dataY)[:-k]
        #self.dataTargetZ = np.asarray(self.dataZ)[k:] - np.asarray(self.dataZ)[:-k]
        
        #time, vx, vy, vz, yaw, pitch, roll, altitude, x, y, z, targetX, targetY, targetZ, w1, w2, w3, w4
        #time, yaw, pitch, roll, altitude, x, y, z, targetX, targetY, targetZ, w1, w2, w3, w4
        #time, yaw, pitch, roll, x, y, z, targetX, targetY, targetZ, w1, w2, w3, w4
        """
        self.data = np.column_stack((np.asarray(self.dataTimeDiffs)[:-k], 
                                        # (np.asarray(self.dataVX)/Vscale)[:-k], (np.asarray(self.dataVY)/Vscale)[:-k], (np.asarray(self.dataVZ)/Vscale)[:-k],
                                         (np.asarray(self.dataYaw)[:-k])/Yaw_scale,
                                     np.asarray(self.dataPitch)[:-k], np.asarray(self.dataRoll)[:-k], 
                                     np.asarray(self.dataAltitude)[:-k],
                                     np.asarray(self.dataX)[:-k], np.asarray(self.dataY)[:-k], np.asarray(self.dataZ)[:-k],
                                     np.asarray(self.dataTargetX), np.asarray(self.dataTargetY), np.asarray(self.dataTargetZ),
                                     np.asarray(self.dataw1)[:-k], np.asarray(self.dataw2)[:-k], np.asarray(self.dataw3)[:-k], np.asarray(self.dataw4)[:-k]))
        """
        self.data = np.asarray(self.dataTimeDiffs)[:-k]
        if self.load_dV:
            self.data = np.column_stack((self.data, (np.asarray(self.dataVX)/Vscale)[:-k], (np.asarray(self.dataVY)/Vscale)[:-k], (np.asarray(self.dataVZ)/Vscale)[:-k],))
        self.data = np.column_stack((self.data, (np.asarray(self.dataYaw)[:-k])/Yaw_scale,
                                     np.asarray(self.dataPitch)[:-k], np.asarray(self.dataRoll)[:-k]))
        if self.load_altitude:
            self.data = np.column_stack((self.data, np.asarray(self.dataAltitude)[:-k]))
        if self.load_xyz:
            self.data = np.column_stack((self.data, np.asarray(self.dataX)[:-k], np.asarray(self.dataY)[:-k], np.asarray(self.dataZ)[:-k],
                                     np.asarray(self.dataTargetX), np.asarray(self.dataTargetY), np.asarray(self.dataTargetZ)))
        self.data = np.column_stack((self.data, np.asarray(self.dataw1)[:-k], np.asarray(self.dataw2)[:-k], 
                                     np.asarray(self.dataw3)[:-k], np.asarray(self.dataw4)[:-k]))      
        """
        if self.containsTarget:
            #Relative target
            self.dataTargetX = np.asarray(self.dataTargetX) - np.asarray(self.dataX)
            self.dataTargetY = np.asarray(self.dataTargetY) - np.asarray(self.dataY)
            self.dataTargetZ = np.asarray(self.dataTargetZ) - np.asarray(self.dataZ)
            
            #time, vx, vy, vz, yaw, pitch, roll, altitude, x, y, z, targetX, targetY, targetZ, w1, w2, w3, w4
            self.data = np.column_stack((np.asarray(self.dataTimeDiffs), 
                                         np.asarray(self.dataVX)/Vscale, np.asarray(self.dataVY)/Vscale, np.asarray(self.dataVZ)/Vscale,
                                         (np.asarray(self.dataYaw))/Yaw_scale,
                                     np.asarray(self.dataPitch), np.asarray(self.dataRoll), np.asarray(self.dataAltitude),
                                     np.asarray(self.dataX), np.asarray(self.dataY), np.asarray(self.dataZ),
                                     np.asarray(self.dataTargetX), np.asarray(self.dataTargetY), np.asarray(self.dataTargetZ),
                                     np.asarray(self.dataw1), np.asarray(self.dataw2), np.asarray(self.dataw3), np.asarray(self.dataw4)))
            #self.w_columns = np.arange(11, 15) 
            self.w_columns = np.arange(14, 18) 
        else:
            #time, vx, vy, vz, yaw, pitch, roll, altitude, x, y, z, w1, w2, w3, w4
            self.data = np.column_stack((np.asarray(self.dataTimeDiffs),
                    np.asarray(self.dataVX)/Vscale, np.asarray(self.dataVY)/Vscale, np.asarray(self.dataVZ)/Vscale, 
                    (np.asarray(self.dataYaw))/Yaw_scale, np.asarray(self.dataPitch), np.asarray(self.dataRoll), np.asarray(self.dataAltitude),
                    np.asarray(self.dataX), np.asarray(self.dataY), np.asarray(self.dataZ),
                    np.asarray(self.dataw1), np.asarray(self.dataw2), np.asarray(self.dataw3), np.asarray(self.dataw4)))
            self.w_columns = np.arange(8, 12)
        """
        #self.data = np.hstack((self.data, self.dataAltitude, 1))
        #self.data = np.hstack((self.data, self.dataYaw, 2))
        
        #Ignore launch~ first 5 seconds and the last seconds in the end
        cutoff = 50
        self.data = self.data[cutoff:-cutoff,:]
        #print self.data.shape
        
if __name__ == '__main__':
    flight_data = FlightData('flight_data/flight_random_points_with_target/flight_Wed_06_Feb_2013_16_07_34_AllData')
    
    #print len(data.dataZ), len(data.dataCommand), len(data.dataw1), len(data.dataPitch)
    #print len(flight_data.dataTime)
