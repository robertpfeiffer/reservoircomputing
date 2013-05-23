import json
import numpy
import select
import socket
import struct
import time
import sys
import os 

#UDP_IP = "127.0.0.1"
#UDP_PORT = 5005
#
#sock = socket.socket(socket.AF_INET, # Internet
#                     socket.SOCK_DGRAM) # UDP
#sock.bind((UDP_IP, UDP_PORT))

class RecordData(object):
    stop=False
    
    
    def __init__(self):
        print 'Lets go !'
        self.duration=0
    
    def setStop(self,value):
        self.stop=value
            
    def decode_navdata(self, packet):
        """Decode a navdata packet."""
        offset = 0
        _ =  struct.unpack_from("IIII", packet, offset)
        drone_state = dict()
        drone_state['fly_mask']             = _[1]       & 1 # FLY MASK : (0) ardrone is landed, (1) ardrone is flying
        drone_state['video_mask']           = _[1] >>  1 & 1 # VIDEO MASK : (0) video disable, (1) video enable
        drone_state['vision_mask']          = _[1] >>  2 & 1 # VISION MASK : (0) vision disable, (1) vision enable */
        drone_state['control_mask']         = _[1] >>  3 & 1 # CONTROL ALGO (0) euler angles control, (1) angular speed control */
        drone_state['altitude_mask']        = _[1] >>  4 & 1 # ALTITUDE CONTROL ALGO : (0) altitude control inactive (1) altitude control active */
        drone_state['user_feedback_start']  = _[1] >>  5 & 1 # USER feedback : Start button state */
        drone_state['command_mask']         = _[1] >>  6 & 1 # Control command ACK : (0) None, (1) one received */
        drone_state['fw_file_mask']         = _[1] >>  7 & 1 # Firmware file is good (1) */
        drone_state['fw_ver_mask']          = _[1] >>  8 & 1 # Firmware update is newer (1) */
        drone_state['fw_upd_mask']          = _[1] >>  9 & 1 # Firmware update is ongoing (1) */
        drone_state['navdata_demo_mask']    = _[1] >> 10 & 1 # Navdata demo : (0) All navdata, (1) only navdata demo */
        drone_state['navdata_bootstrap']    = _[1] >> 11 & 1 # Navdata bootstrap : (0) options sent in all or demo mode, (1) no navdata options sent */
        drone_state['motors_mask']          = _[1] >> 12 & 1 # Motor status : (0) Ok, (1) Motors problem */
        drone_state['com_lost_mask']        = _[1] >> 13 & 1 # Communication lost : (1) com problem, (0) Com is ok */
        drone_state['vbat_low']             = _[1] >> 15 & 1 # VBat low : (1) too low, (0) Ok */
        drone_state['user_el']              = _[1] >> 16 & 1 # User Emergency Landing : (1) User EL is ON, (0) User EL is OFF*/
        drone_state['timer_elapsed']        = _[1] >> 17 & 1 # Timer elapsed : (1) elapsed, (0) not elapsed */
        drone_state['angles_out_of_range']  = _[1] >> 19 & 1 # Angles : (0) Ok, (1) out of range */
        drone_state['ultrasound_mask']      = _[1] >> 21 & 1 # Ultrasonic sensor : (0) Ok, (1) deaf */
        drone_state['cutout_mask']          = _[1] >> 22 & 1 # Cutout system detection : (0) Not detected, (1) detected */
        drone_state['pic_version_mask']     = _[1] >> 23 & 1 # PIC Version number OK : (0) a bad version number, (1) version number is OK */
        drone_state['atcodec_thread_on']    = _[1] >> 24 & 1 # ATCodec thread ON : (0) thread OFF (1) thread ON */
        drone_state['navdata_thread_on']    = _[1] >> 25 & 1 # Navdata thread ON : (0) thread OFF (1) thread ON */
        drone_state['video_thread_on']      = _[1] >> 26 & 1 # Video thread ON : (0) thread OFF (1) thread ON */
        drone_state['acq_thread_on']        = _[1] >> 27 & 1 # Acquisition thread ON : (0) thread OFF (1) thread ON */
        drone_state['ctrl_watchdog_mask']   = _[1] >> 28 & 1 # CTRL watchdog : (1) delay in control execution (> 5ms), (0) control is well scheduled */
        drone_state['adc_watchdog_mask']    = _[1] >> 29 & 1 # ADC Watchdog : (1) delay in uart2 dsr (> 5ms), (0) uart2 is good */
        drone_state['com_watchdog_mask']    = _[1] >> 30 & 1 # Communication Watchdog : (1) com problem, (0) Com is ok */
        drone_state['emergency_mask']       = _[1] >> 31 & 1 # Emergency landing : (0) no emergency, (1) emergency */
        data = dict()
        data['drone_state'] = drone_state
        data['header'] = _[0]
        data['seq_nr'] = _[2]
        data['vision_flag'] = _[3]
        offset += struct.calcsize("IIII")
        while 1:
            try:
                id_nr, size =  struct.unpack_from("HH", packet, offset)
                offset += struct.calcsize("HH")
            except struct.error:
                break
            values = []
            for i in range(size-struct.calcsize("HH")):
                values.append(struct.unpack_from("c", packet, offset)[0])
                offset += struct.calcsize("c")
            # navdata_tag_t in navdata-common.h
            if id_nr == 0:
                values = struct.unpack_from("IIfffIfffI", "".join(values))
                values = dict(zip(['ctrl_state', 'battery', 'theta', 'phi', 'psi', 'altitude', 'vx', 'vy', 'vz', 'num_frames'], values))
                # convert the millidegrees into degrees and round to int, as they
                # are not so precise anyways
                for i in 'theta', 'phi', 'psi':
                    values[i] = int(values[i] / 1000)
                    #values[i] /= 1000
            data[id_nr] = values
        return data
    
        
    def run(self):    
        
        print '_duration: ' + str(self.duration)
        DRONE_PORT=5554
        TRACKER_PORT = 10122
        COMMU_PORT = 10124
        TRACKER_IP = '127.0.0.1'
        DRONE_IP="192.168.1.1"
        MY_IP = "192.168.1.2"#"127.0.0.1"
        
        COMMANDER_IP = "127.0.0.1"
        
        drone_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        drone_socket.bind( (MY_IP, DRONE_PORT) )
        drone_socket.sendto("\x01\x00\x00\x00", (DRONE_IP, 5554))
        drone_socket.setblocking(0)
        
        tracker_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #for receiving tracker data
        tracker_socket.bind((TRACKER_IP, TRACKER_PORT)) #listens on this port
        tracker_socket.setblocking(0)
        
        commu_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #for receiving current command data
        commu_socket.bind((COMMANDER_IP, COMMU_PORT)) #listens on this port
        commu_socket.setblocking(0)
        
        COMMANDER_PORT2 = 10126
        SOCK_COMMANDER = socket.socket(socket.AF_INET, # Internet
                         socket.SOCK_DGRAM) # UDP
#        SOCK_COMMANDER.bind(('',COMMANDER_PORT2)) #this is a server
#        SOCK_COMMANDER.setblocking(0)
        
        #video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #video_socket.setblocking(1) #orig 0
        #video_socket.bind(('', libardrone.ARDRONE_VIDEO_PORT)) #kristoffer's change, original ''
        #video_socket.sendto("\x01\x00\x00\x00", ('192.168.1.1', libardrone.ARDRONE_VIDEO_PORT))
        
        readlist = [tracker_socket,drone_socket,commu_socket]
        
        while True:
            (sread, swrite, sexc) =  select.select(readlist, [], [],1.0 )
            if drone_socket in sread:
                addr=None
                data=None
                while 1:
                    try:
                        data, addr = drone_socket.recvfrom(65535)
                    except IOError:
                        # we consumed every packet from the socket and
                        # continue with the last one
                        break                   
                assert addr == (DRONE_IP, DRONE_PORT)
                decodedData=self.decode_navdata(data)
    #            navdata = decodedData
                print 'received drone data! '
                try:
                    print 'Battery0: ' + str(decodedData[0]['battery'])
                except:
                    print 'incomplete drone data' 
                    drone_socket.sendto("\x01\x00\x00\x00", (DRONE_IP, 5554))
                    time.sleep(.5)
                break
            else:
                print 'nothing received from the drone within one second.'
                
                print 'trying it again...'
                drone_socket.sendto("\x01\x00\x00\x00", (DRONE_IP, 5554))
                time.sleep(.5)
        
        start_time=time.clock()
        cycles=0L #elapsed timesteps
        freq=10   # data points per second
        
        DATA_LENGTH = (int(self.duration))*freq
        dims='battery','vx','vy','vz','psi','theta','phi','altitude'
        
        all_measurements = []
        
        navdata=None
        tracker_data=None
        command_data=None

        elapsed=(time.clock()-start_time)
        
        last_commander_time = time.time()
        
        oldBattery=123
        autoStat=0
        otherStat=0
        stopp=1
        try:
            while True:
                retry=True
                (sread, swrite, sexc) =  select.select(readlist, [], [], 0.01 )
                if commu_socket in sread:
                    while 1:
                        try:
                            command_data, addr = commu_socket.recvfrom(1024)
                        except IOError:
                            # we consumed every packet from the socket and
                            # continue with the last one
                            break   
                    print 'commu_socket at ' + str(addr)
                    if command_data[:4]=='auto':
                        autoStat=autoStat+1
                    else:
                        if command_data<>'None':
                            otherStat=otherStat+1                    
                if str(command_data)[:3]=='end':
                    break
                
                if drone_socket in sread: 
                    while 1:
                        try:
                            data, addr = drone_socket.recvfrom(65535)
                        except IOError:
                            # we consumed every packet from the socket and
                            # continue with the last one
                            break                
                    
                    print 'drone socket at ' + str(addr)
                    navdata=self.decode_navdata(data)
                
                if tracker_socket in sread:
                    while 1:
                        try:
                            data, addr = tracker_socket.recvfrom(1024)
                        except IOError:
                            # we consumed every packet from the socket and
                            # continue with the last one
                            break                
                    
                    print 'tracker socket at ' + str(addr)
                    tracker_data=data
    
                    
                elapsed=(time.clock()-start_time)
    #            if cycles<int(elapsed*freq):
                if navdata and tracker_data and 0 in navdata:
                    yDroneStart=tracker_data.find('y_drone')
                    zDroneStart=tracker_data.find('z_drone')
                    zDroneEnd=tracker_data.find('#')
                    x_drone=float(tracker_data[7:yDroneStart])
                    y_drone=float(tracker_data[yDroneStart+7:zDroneStart])
                    z_drone=float(tracker_data[zDroneStart+7:zDroneEnd])

                    yLkwStart=tracker_data.find('y_lkw')
                    zLkwStart=tracker_data.find('z_lkw')
                    zLkwEnd=tracker_data.find('!')
                    x_lkw=float(tracker_data[zDroneEnd+6:yLkwStart])
                    y_lkw=float(tracker_data[yLkwStart+5:zLkwStart])
                    z_lkw=float(tracker_data[zLkwStart+5:zLkwEnd])


                    
                    if navdata[0]['battery']<>oldBattery:
                        print 'Battery1: ' + str(navdata[0]['battery'])
                        oldBattery=navdata[0]['battery']
                    measurement=[navdata[0][a] for a in dims]+[x_drone]+[y_drone]+[z_drone]+ [x_lkw]+[y_lkw]+[z_lkw]+[command_data] + [time.time()]
                    print measurement
                    all_measurements=all_measurements + [measurement]
                    
                    navdata=None
                    tracker_data=None
                    
                if time.time()-last_commander_time > 0.1:
                    last_commander_time = time.time()
                    msg=''
                    for i in measurement:
                        msg = msg + str(i) + '#' 
                    SOCK_COMMANDER.sendto(msg, (COMMANDER_IP, COMMANDER_PORT2))
                

                if stopp==1234:
                    break #manual stop point
        except:
            print 'loop error:', sys.exc_info()[0], sys.exc_info()[1]
            #raise

        print 'save file...'
        
       

        print elapsed
        serialised=''
#        serialised=str(all_measurements)
        
        for m in all_measurements:
            serialised=serialised + str(m) + '\n'
        
        if float((autoStat+otherStat))<>0:
            stat=autoStat/float((autoStat+otherStat))
        else:
            stat=1.0
        serialised = serialised + 'Statistics: \n TotalFlightCommands: ' + str(autoStat+otherStat) + '\n Correcting: ' + str(autoStat) +'\n Regular Command: ' + str(otherStat) + '\n Fraction Correcting: ' + str(stat)    
        
        tmpstmp=time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        f=open("flight_"+tmpstmp+'_AllData',"w")
        
        f.write(serialised)
        f.close()
       
        drone_socket.close()
        
    def stop(self):
        self.duration=0

if __name__ == '__main__':
    print 'start recording...'
    rec_data = RecordData()
    rec_data.duration=0
#    try:
#        rec_data.duration=int(sys.argv[1])
#    except:
#        print 'no argument'
#    if rec_data.duration==0:
#        rec_data.duration=9999
    rec_data.run()
