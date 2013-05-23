""" ---------------------------------------------------------------------- #
                AR Drone Command and Track Script
                Version 2.0
#   ------------------------------------------------------------------ """

import ardrone.ArdroneCommander as ArdroneCommander
import math
import os
import pygame
import random
import select
import socket
import struct
import sys
import time
import reservoircomputing.esn_persistence as esn_persistence

from drone_esn import *

MARGIN = 0.8
# Trackable area [xmin, xmax, ymin, ymax, zmin, zmax]
TRACKING_AREA = [-3.5, 3.5, 0.6, 1.9, -3.5, 3.5]

x_target_margin = 0.2 #0.4
y_target_margin = 0.1 #0.1
z_target_margin = 0.2 #0.4

class CommandAndTrack(object):
    
    """ -------------------------------------------------------------------- #
                    Initialize Drone and communication
    #   ------------------------------------------------------------------ """
    def __init__(self):
        self.log1 = '' # log1 structure: Timestamp, Search time, target point, current position of drone, boolean out of tracking area, boolean not yet at target point, battery
        self.log2 = '' #Timestamp. Target reached or not. Searchtime. Target. Battery.   
        
        tmpstmp=time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        
        self.log1_filename = 'statistics_1_'+tmpstmp+'.txt'        
        self.log2_filename = 'statistics_2_'+tmpstmp+'.txt'
        
        
        
        print 'init...'
        
        self.ESN_control = True
#        self.ESN_control = False
        if self.ESN_control:
            self.with_esn = True
            self.drone_esn = DroneESN()
            print 'Successfully generated DroneESN()'
        else:
            self.with_esn = False
        self.ESN_control = False #always false
        
        self.outoftrackingarea=False
        
        pygame.init()
        self.interval = 0.03
        self.command_list = []

        
        self.createCommandList()

        print 'communication...'
        self.UDP_IP_communication = "127.0.0.1"
        self.UDP_PORT_communication = 10124
        self.sock_communication = socket.socket(socket.AF_INET,     # Internet
                                                socket.SOCK_DGRAM)  # UDP            

        print 'commander..'
        self.commander_sock = socket.socket(socket.AF_INET,         # Internet
                                            socket.SOCK_DGRAM)      # UDP
        self.commander_sock.setblocking(0)
        self.commander_sock.bind( ('', 10126) )

        self.readlist = [self.commander_sock]
        self.yaw = '181'
        self.x = '0'
        self.y = '0'
        self.z = '0'
        
        self.pitch='12345'
        self.roll = '12345'
        self.timestamp = str(time.time())
        
        self.vx='0'
        self.vy='0'
        self.vz='0'
        
        self.altitude='0'
        
        self.targetPoint = [1.0,1.0,1.0]
        self.mustFlyToCenter = True
        self.targetCount = -1
        self.targetReached = -1
        self.targetStartSearch = 0
        self.targetReachingDuration = 0
                      
        self.all_data = []
        
        self.overTargetCount=0
        
        self.battery = 123

        self.ardrone_commander = ArdroneCommander.ArdroneCommander()
        self.ardrone_commander.takeOff()

        self.flightLoop()

        self.ardrone_commander.land()
        try:
            if self.with_esn:
                self.drone_esn.save_echo()
                #esn_persistence.save_object(self.drone_esn, "esn_drone.txt")
        except:
            print 'Error while saving data: ',sys.exc_info()[0], sys.exc_info()[1]


    def saveLogs(self):
        
        f=open(self.log1_filename,"a")
        f.write(str(self.log1))
        f.close()
        self.log1=''

        f=open(self.log2_filename,"a")
        f.write(str(self.log2))
        f.close()
        self.log2=''
        

    """ -------------------------------------------------------------------- #
                    Drone main flight handling procedure
                    
        * Drone works through the flight-command-list
        * If drone leaves the tracking area, trivial autopilot is called
          to push it back
    #   ------------------------------------------------------------------ """
    def flightLoop(self):

        last_time = time.time()
        search_time=0
        print 'now run until command list is empty'

        while self.command_list<>[]:
            if time.time()-last_time > self.interval:
                last_time = time.time()
                
                #tmpstmp=time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
                tmpstmp = str(last_time)
                # log1 structure: Timestamp, Search time, target point, current position of drone, boolean out of tracking area, boolean not yet at target point, battery
                self.log1=self.log1 + str([str(tmpstmp), str(search_time)[:5], str(self.targetPoint[0])[:5] +' '+ str(self.targetPoint[1])[:5] +' '+ str(self.targetPoint[2])[:5], str(self.x)[:5] + ' '+ str(self.y)[:5] + ' '+ str(self.z)[:5], str(self.outoftrackingarea), str(self.mustFlyToCenter), str(self.battery)])  + '\n' 
                
                
                
                #print "command_list[0]", str(self.command_list[0])
                if self.command_list[0]>0: #self.command_list[0] ist die zeit, die schon im aktuellen ziel verbracht wurde
                    if self.targetStartSearch == 0:
                        search_time = 0
                    else:
                        search_time = time.time() - self.targetStartSearch
                    if self.command_list[0] == 'target':
                        print 'target reached. '
                        self.log2=self.log2 + str([str(tmpstmp), 'Target     reached after: ', str(search_time)[:5], str(self.targetPoint[0])[:5] +' '+ str(self.targetPoint[1])[:5] +' '+ str(self.targetPoint[2])[:5], str(self.battery)]) + '\n'
                        self.saveLogs() 
                        
                        print 'Targets reached: ' + str(self.targetReached) + '/' + str(self.targetCount) + '\t NEW target at:' + str(self.command_list[1])
                        self.overTargetCount=0
                        self.targetPoint = self.command_list[1]
                        self.command_list = self.command_list[2:]
                        self.targetStartSearch = time.time()
                        self.getFreshData()
                        self.stayInside()
                        self.targetCount += 1
                        self.targetReached += 1
                        if self.with_esn and self.targetCount > 0:
                            self.ESN_control=True
                    elif search_time > 30:
                        if self.command_list[1]<>'autoleft' and self.command_list[1]<>'autoright':
                            self.targetReached -= 1
                            print 'target not reached'
                            self.log2=self.log2 + str([str(tmpstmp), 'Target not reached after: ', str(search_time)[:5], str(self.targetPoint[0])[:5] +' '+ str(self.targetPoint[1])[:5] +' '+ str(self.targetPoint[2])[:5], str(self.battery)]) + '\n'
                            self.saveLogs()
                            
                        if self.command_list[1]<>'winkel':
                            self.command_list = self.command_list[2:]
                        else:
                            self.command_list = self.command_list[6:]
                        print 'Targets NOT reached in 10s' #+ str(self.targetCount) + '\t NEW target at:' + str(self.command_list[1])
                    else:
                        targetstr = str(self.targetPoint[0])[:4] +' '+ str(self.targetPoint[1])[:4] +' '+ str(self.targetPoint[2])[:4]   
                        print 'Searchtime: ' + str(search_time)[:3] + 's Targets reached: ' + str(self.targetReached) + '/' + str(self.targetCount) + '\t Target at:' + targetstr + '\t Current Pos.: '+ str(self.x)[:4] + ' '+ str(self.y)[:4] + ' '+ str(self.z)[:4] 
                        self.command_list[0] = float(self.command_list[0])-self.interval
                        
                        if self.command_list[1]<>'end':
                            self.getFreshData()
                            self.stayInside()
                                                 
                        if int(self.battery)<40 and self.command_list[1]<>'end':
                            self.command_list = [1,'end']
                        if self.command_list[0]<>'target':
                            #print 'self.command_list[0]<>target'
                            if self.command_list[1]<>'autowinkel':
                                print 'command: ' + str(self.command_list[1])
                            if self.command_list[1]=='winkel':
                                self.ardrone_commander.w1=self.command_list[2]
                                self.ardrone_commander.w2=self.command_list[3]
                                self.ardrone_commander.w3=self.command_list[4]
                                self.ardrone_commander.w4=self.command_list[5]
                                print 'winkel ' + str(self.command_list[2]) + ' ' + str(self.command_list[3]) + ' ' + str(self.command_list[4]) + ' '
                                
                            self.ardrone_commander.command(self.command_list[1])
                            self.saveData()
                else:
                    # If the ordered time interval for the current movement is over/target is reached,
                    # take the next movement/target command.
                    if self.command_list[1]<>'winkel':
                        self.command_list = self.command_list[2:]
                    else:
                        self.command_list = self.command_list[6:]

    """ -------------------------------------------------------------------- #
                    Updates and parses the data send by the drone
    #   ------------------------------------------------------------------ """
    def getFreshData(self):
        self.readlist = [self.commander_sock]
        (sread, swrite, sexc) =  select.select(self.readlist, [], [], 0.01)
        if self.commander_sock in sread:
            while 1:
                try:
                    msg, addr = self.commander_sock.recvfrom(1024)
                except IOError:
                    # we consumed every packet from the socket and
                    # continue with the last one
                    break                     
#            print 'received from: ' + str(addr) + ' this message: ' + str(msg)
            try:
                # Parse the message to extract all parameter values
                temp=str.find(msg, '#')
                self.battery=msg[:temp]
                msg=msg[temp+1:]
                
                temp=str.find(msg, '#')
                self.vx=msg[:temp]
                msg=msg[temp+1:]
                
                temp=str.find(msg, '#')
                self.vy=msg[:temp]
                msg=msg[temp+1:]
    
                temp=str.find(msg, '#')
                self.vz=msg[:temp]
                msg=msg[temp+1:]
    
                temp=str.find(msg, '#')
                self.yaw=msg[:temp]
                msg=msg[temp+1:]
    
                temp=str.find(msg, '#')
                self.pitch=msg[:temp]
                msg=msg[temp+1:]
    
                temp=str.find(msg, '#')
                self.roll=msg[:temp]
                msg=msg[temp+1:]
    
                temp=str.find(msg, '#')
                self.altitude=msg[:temp]
                msg=msg[temp+1:]
    
                temp=str.find(msg, '#')
                self.x=msg[:temp]
                msg=msg[temp+1:]
    
                temp=str.find(msg, '#')
                self.y=msg[:temp]
                msg=msg[temp+1:]

                temp=str.find(msg, '#')
                self.z=msg[:temp]
                msg=msg[temp+1:]

                temp=str.find(msg, '#')
                command=msg[:temp]
                msg=msg[temp+1:]  

                temp=str.find(msg, '#')
                self.timestamp=msg[:temp]
                msg=msg[temp+1:]              
            except:
                print 'message decoding error'

    """ -------------------------------------------------------------------- #
            Drone turns to face the north pole direction until success
    #   ------------------------------------------------------------------ """
    def faceNorth(self):
        if self.command_list[1]<>'autoturnleft' and self.command_list[1]<>'autoturnright':
            if int(self.yaw) > 10:
                self.command_list = [0.1,'autoturnleft'] + self.command_list
                print self.command_list
            elif int(self.yaw) < -10:
                self.command_list = [0.1,'autoturnright'] + self.command_list
            else:
                print 'successfully faced north'

    def flyTo(self,x,z):
        pass

    """ -------------------------------------------------------------------- #
        Drone checks if it is outside a defined area - the Home/Save zone.
        If so it initiates a correcting movement command to get into the zone.
    #   ------------------------------------------------------------------ """
    def stayInside(self):
        
##        if self.mustFlyToCenter == True:
            
        margin = 1.5
        #supermarket trackable area: x -2 3, z -2 4
        
        #tracking area - if outside, then return to self.targetPoint
        xmin,xmax,ymin,ymax,zmin,zmax = TRACKING_AREA
        
        x = self.x
        y = self.y
        z = self.z
        
        self.currentTarget=self.targetPoint
        
        original_ESN_control=self.ESN_control
        
        self.outoftrackingarea=False
        if float(x)>xmax or float(x)<xmin or float(z)>zmax or float(z)<zmin or float(y)>ymax or float(y)<ymin:
            print 'NO ESN CONTROL!!! OUT OF TRACKING AREA! GOTO 0,1,0'
            self.outoftrackingarea=True
            self.ESN_control=False
            self.currentTarget=[0,1,0]
            
        
        # Home zone / Target zone definition
        
        xmin = self.currentTarget[0] - x_target_margin
        xmax = self.currentTarget[0] + x_target_margin
        ymin = self.currentTarget[1] - y_target_margin
        ymax = self.currentTarget[1] + x_target_margin
        zmin = self.currentTarget[2] - z_target_margin
        zmax = self.currentTarget[2] + z_target_margin
        
        
        w1 = 0
        w2 = 0
        w3 = 0
        w4 = 0

        if original_ESN_control:
            ESN_param=[float(self.yaw), float(self.x_drone), float(self.y_drone), float(self.z_drone), float(self.targetPoint[0]), float(self.targetPoint[1]), float(self.targetPoint[2]) ]
            wwww = self.drone_esn.compute(ESN_param)
        
        # Check if drone is inside x and z range of target area
        if float(x)>xmax or float(x)<xmin or float(z)>zmax or float(z)<zmin or float(y)>ymax or float(y)<ymin:
            
            self.mustFlyToCenter=True

            if self.ESN_control:
                
                print 'ESN correcting...'
                              
                wwww = wwww[0]
                w1 = wwww[0]
                w2 = wwww[1]
                w3 = wwww[2]
                w4 = wwww[3]

                print 'w1: ' + str(w1)
                print 'w2: ' + str(w2)
                print 'w3: ' + str(w3)
                print 'w4: ' + str(w4)
                print ' ---------- '
                
                if w1>0.3:
                    w1=0.3
                if w1<-0.3:
                    w1=-0.3
                if w2>0.3:
                    w2=0.3
                if w2<-0.3:
                    w2=-0.3
                if w3>0.4:
                    w1=0.4
                if w3<-0.1:
                    w3=-0.1
                if w4>0.3:
                    w4=0.3
                if w4<-0.3:
                    w4=-0.3
                
                if float(y)>ymax:
                    w3 = -0.1
                if float(y)<ymin:
                    w3 = 0.4
                    

                
            else:
                #print '#not at target point'
                
                self.targetReachingDuration = time.time() - self.targetStartSearch
                
                # speed_factor = ((float(x)**2+float(z)**2)**0.5)/7*0.5
                # max deviation = 5 m, euklidian distance max sqrt(50), therefore normalized by 7, 0.5 is maximum speed factor
                if self.targetReachingDuration > 1:
                    speed_factor = ( (((float(x)-self.currentTarget[0])**2 + (float(z)-self.currentTarget[2])**2)**0.5)/7 * 0.5 ) / self.targetReachingDuration
                else:
                    speed_factor = (((float(x)-self.currentTarget[0])**2 + (float(z)-self.currentTarget[2])**2)**0.5)/7 * 0.5
                    
                if speed_factor < 0.1: speed_factor = 0.1
    #            print 'Distance to goal, x: ' + str(float(x)-self.currentTarget[0]) + ', z: ' + str(float(z)-self.currentTarget[2])
    
                # Case differentiation where the drone is located relative to the origin of the target / Home zone.
                # Calculation of the beta parameter which is used for the correcting movement.
                if float(x)-self.currentTarget[0]>0 and float(z)-self.currentTarget[2]>0:
                    beta=math.degrees(math.atan((float(x)-self.currentTarget[0])/(float(z)-self.currentTarget[2])))
                if float(x)-self.currentTarget[0]>0 and float(z)-self.currentTarget[2]<0:
                    beta=90+math.degrees(math.atan(-(float(z)-self.currentTarget[2])/(float(x)-self.currentTarget[0])))
                if float(x)-self.currentTarget[0]<0 and float(z)-self.currentTarget[2]<0:
                    beta=-(90+math.degrees(math.atan(-(float(z)-self.currentTarget[2])/-(float(x)-self.currentTarget[0]))))
                if float(x)-self.currentTarget[0]<0 and float(z)-self.currentTarget[2]>0:
                    beta=-(math.degrees(math.atan(-(float(x)-self.currentTarget[0])/(float(z)-self.currentTarget[2]))))
                    
                # Set w1 and w2 by our movement correction formula
                w1 = (math.sin( math.radians(float(self.yaw)) + math.radians(-beta) )) * speed_factor
                w2 = (math.cos( math.radians(float(self.yaw)) + math.radians(-beta) )) * speed_factor
                #print 'stayinside w1: ' + str(w1) + ', w2: ' + str(w2)
                
                if float(y)>ymax:
                    w3 = -0.1
                if float(y)<ymin:
                    w3 = 0.4
        else:
            self.mustFlyToCenter = False
            
           
#            print w1
#            print w2

        # Check if drone is inside y range of target area (altitude)
##        if float(y)<0.7:
##            self.flexibleymin = 1.8
##            w3 = 0.4
##        elif float(y)<1.8 and self.flexibleymin == 1.8:
##            w3 = 0.4
##        else:
##            self.flexibleymin = 0.7
##            w3 = 0.0 



   
#        msg = self.command_list[1] + ' w1,w2,w3,w4= ' + str(self.ardrone_commander.w1) + ', ' + str(self.ardrone_commander.w2) + ', ' + str(self.ardrone_commander.w3) + ', ' + str(self.ardrone_commander.w4)
#        print str(msg)

        if self.command_list[1] == 'autowinkel':
            if w1==0 and w2==0 and w3==0 and w4==0:
                self.command_list = self.command_list[2:]
        else:
            self.command_list = [1000, 'autowinkel'] + self.command_list
            msg = self.command_list[1] + ' w1,w2,w3,w4= ' + str(self.ardrone_commander.w1) + ', ' + str(self.ardrone_commander.w2) + ', ' + str(self.ardrone_commander.w3) + ', ' + str(self.ardrone_commander.w4)
            print str(msg)
        
###     #add noise to the w
###        noise_factor = 0.3
###     
###        noise1= random.random()*noise_factor-0.5*noise_factor
###        noise2= random.random()*noise_factor-0.5*noise_factor
###        noise3= random.random()*noise_factor-0.5*noise_factor
###
###        w1=w1-noise1
###        w2=w2-noise2
###        w3=w3-noise3
        

        # Send adjusted angle parameters to drone
        self.ardrone_commander.w1 = w1
        self.ardrone_commander.w2 = w2
        self.ardrone_commander.w3 = w3
        self.ardrone_commander.w4 = w4
        
        self.ESN_control = original_ESN_control
        
        
#        if w1==0 and w2==0 and w3==0 and w4==0:
#            if self.command_list[1] == 'autowinkel':
#                self.command_list = self.command_list[2:]
#        else:
#            if self.command_list[1] <> 'autowinkel':
#                self.command_list = [1000, 'autowinkel'] + self.command_list
#                msg = self.command_list[1] + ' w1,w2,w3,w4= ' + str(self.ardrone_commander.w1) + ', ' + str(self.ardrone_commander.w2) + ', ' + str(self.ardrone_commander.w3) + ', ' + str(self.ardrone_commander.w4)
#                print str(msg)

    """ -------------------------------------------------------------------- #
        Function sends the current movement command with its parameter
        values to DroneSocketTest, which saves it with the rest of the data.
    #   ------------------------------------------------------------------ """
    def saveData(self):
        if self.command_list[1]=='autowinkel':
            msg = self.command_list[1] + ' w1,w2,w3,w4= ' + str(self.ardrone_commander.w1) + ', ' + str(self.ardrone_commander.w2) + ', ' + str(self.ardrone_commander.w3) + ', ' + str(self.ardrone_commander.w4) + '; target: ' + str(self.targetPoint)
            self.sock_communication.sendto(msg, (self.UDP_IP_communication, self.UDP_PORT_communication))
        else:
            self.sock_communication.sendto(self.command_list[1] + '; target: ' + str(self.targetPoint), (self.UDP_IP_communication, self.UDP_PORT_communication))

    def createCommandList(self):
        self.command_list=[4,'hover']
        self.command_list=['target', [2, 1, 2]]
        
        #trackable area: x -2 3, z -2 4
        
        for i in range(1000):
###            #from A [-1, 1.2, 3] to B [2, 1.2, 3], constant yaw, no ventilator, until battery empty
###            self.command_list = self.command_list + ['target', [-1, 1.2, -1],'target', [-1, 0.7, 3],'target', [2, 1.7, 2]]
###            

#            from A [-1, 1, 0] to B [0.5, 1, 0], no ventilator, until battery empty
#######            for j in range(5):
#######                self.command_list = self.command_list + ['target', [1, 1, -1],'target', [-0.5, 1, 0.5]]
#######            self.command_list = self.command_list + [1, 'hover']  
#######            
###            #from A [-1, 1.2, 3] to B [2, 1.2, 3], every 5th time changing yaw, no ventilator, until battery empty
###            for j in range(3):
###                self.command_list = self.command_list + ['target', [-1, 1.2, 3],'target', [2, 1.2, 3]]
###            self.command_list = self.command_list + [3, 'turnleft']              
            
            # random targets inside the trackable area
#            rx=random.randint(-2,3)
#            rz=random.randint(-2,4)
#            self.command_list = self.command_list + ['target', [rx, 1.2, rz]]

            # random targets inside the trackable area
            for j in range(2):
                x1,x2,y1,y2,z1,z2=TRACKING_AREA
                
                x1+=MARGIN
                z1+=MARGIN
                y1+=MARGIN/10.0
                y2-=MARGIN/10.0
                x2-=MARGIN
                z2-=MARGIN
                
                rx=random.random()*(x2-x1)+x1
                ry=random.random()*(y2-y1)+y1
                rz=random.random()*(z2-z1)+z1
                self.command_list = self.command_list + ['target', [rx, ry, rz]]
            if not self.with_esn:
                self.command_list = self.command_list + [0.8, 'turnleft'] 
            
            # predefined movement rectangle
            #for j in range(1):
            #    self.command_list = self.command_list + ['target', [-1, 1.2, -1], 'target', [-1, 1.2, 3], 'target', [2, 1.2, 3], 'target', [2, 1.2, -1], ]
            #self.command_list = self.command_list + [0.3, 'turnleft'] 
            
            #predefined movement triangle
            #'target', [2, 1.1, 2.3],
#            for j in range(1):
#                self.command_list = self.command_list + ['target', [-1, 1.2, -1], 'target', [0, 1.0, 1], 'target', [1, 1.1, -1]]
#            self.command_list = self.command_list + [0.3, 'turnleft'] 
            

###            # predefined movement HausNikolaus
###            for j in range(1):
###                self.command_list = self.command_list + ['target', [1, 1.5, 3], 1, 'hover', 'target', [-1, 1.5, 3], 1, 'hover', 'target', [1, 1.5, 1], 1, 'hover', 'target', [-1, 1.5, 1], 1, 'hover', 'target', [0, 1.5, -1], 1, 'hover', 'target', [1, 1.5, 1], 1, 'hover', 'target', [1, 1.5, 3], 1, 'hover', 'target', [-1, 1.5, 1], 1, 'hover', 'target', [-1, 1.5, 3], 3, 'hover' ]
###            self.command_list = self.command_list + [0.8, 'turnleft'] 
###            
            
###            # systematically vary the four ws
###            variations=[-0.4, -0.2, 0.0, 0.2, 0.4]
###            for a in range(5):
###                for b in range(5):
###                    for c in range(3):
####                        for d in range(5):
###                        self.command_list = self.command_list + [2, 'winkel', variations[a], variations[b], variations[c]/2, 0.0, 0.6, 'hover' ]
###            
            
            
#            self.command_list=self.command_list+[3,'hover', 2,'forward', 3, 'hover', 2, 'backward', 3, 'hover', 2, 'backward', 3,'hover', 2, 'forward']
#            self.command_list=self.command_list+[1,'forward',2,'hover',1,'backward',2,'hover']  
#            self.command_list=self.command_list+[1,'forward',2,'hover',1,'backward',2,'hover',1,'left',2,'hover',1,'right',2,'hover',1,'turnleft',2,'hover',1,'turnright']
#            self.command_list=self.command_list+[1,'directMotor']
#            self.command_list=self.command_list+[240, 'hover']

            # Scaling octagon
#            self.command_list = self.command_list + ['target', [ 0.0*(i*0.1), 1.0,  3.0*(i*0.1)],
#                                                     'target', [ 2.0*(i*0.1), 0.7, -2.0*(i*0.1)],
#                                                     'target', [ 0.0*(i*0.1), 1.5, -3.0*(i*0.1)],
#                                                     'target', [-2.0*(i*0.1), 0.9, -2.0*(i*0.1)],
#                                                     'target', [-3.0*(i*0.1), 1.2,  0.0*(i*0.1)],
#                                                     'target', [-2.0*(i*0.1), 1.5,  2.0*(i*0.1)],
#                                                     'target', [ 0.0*(i*0.1), 1.3,  3.0*(i*0.1)],
#                                                     'target', [ 2.0*(i*0.1), 0.9,  2.0*(i*0.1)]]

##            # Scaling octagon + scaling altitude
##            self.command_list = self.command_list + ['target', [ 0.0*(i*0.2), 0.8+(i*0.2),  3.0*(i*0.2)],
##                                                     'target', [ 2.0*(i*0.2), 0.8+(i*0.2), -1.0*(i*0.2)],
##                                                     'target', [ 0.0*(i*0.2), 0.8+(i*0.2), -1.0*(i*0.2)],
##                                                     'target', [-1.0*(i*0.2), 0.8+(i*0.2), -1.0*(i*0.2)],
##                                                     'target', [-1.0*(i*0.2), 0.8+(i*0.2),  0.0*(i*0.2)],
##                                                     'target', [-1.0*(i*0.2), 0.8+(i*0.2),  2.0*(i*0.2)],
##                                                     'target', [ 0.0*(i*0.2), 0.8+(i*0.2),  3.0*(i*0.2)],
##                                                     'target', [ 2.0*(i*0.2), 0.8+(i*0.2),  2.0*(i*0.2)]]
        self.command_list = self.command_list + [1,'end']

if __name__ == '__main__':
    CommandAndTrack()
