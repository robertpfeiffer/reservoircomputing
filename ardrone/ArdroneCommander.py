import libardrone
import pygame
import random
import time
import json
import numpy
import select
import socket
import struct




class ArdroneCommander(object):
    def __init__(self):
        print 'Init Lets go!'       
        self.currentCommand=''
        
        self.m1=1
        self.m2=1
        self.m3=1
        self.m4=1
        
        self.w1=0
        self.w2=0
        self.w3=0
        self.w4=0
        """
    w1 -- left-right tilt: float [-1..1] negative: left, positive: right
    w2 -- front-back tilt: float [-1..1] negative: forwards, positive:
        backwards
    w3 -- vertical speed: float [-1..1] negative: go down, positive: rise
    w4 -- angular speed: float [-1..1] negative: spin left, positive: spin 
        right
        """        

        self.flying=False
        self.clock=pygame.time.Clock()
        self.drone = None
        
        self.main()
        
    def fly(self):
#        print 'Command: ' + self.currentCommand
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print "EMERGENCY SHUTDOWN"
                    self.drone.land()
                    self.drone.halt()
                    self.running = False
        if self.currentCommand=='hover':
            self.drone.hover()
            self.currentCommand=''
        elif self.currentCommand=='takeOff':
            self.takeOff()
        elif self.currentCommand=='forward':
            self.drone.move_forward()
        elif self.currentCommand=='backward':
            self.drone.move_backward()
        elif self.currentCommand=='left':
            self.drone.move_left()
        elif self.currentCommand=='right':
            self.drone.move_right()
        elif self.currentCommand=='turnleft':
            self.drone.turn_left()
        elif self.currentCommand=='turnright':
            self.drone.turn_right()
        elif self.currentCommand=='autoturnleft':
            self.drone.turn_left()
        elif self.currentCommand=='autoturnright':
            self.drone.turn_right()

        elif self.currentCommand=='led':
            self.drone.ledFun()
        elif self.currentCommand=='directMotor':
            print 'ardronecommander DM!'
            self.drone.directMotor(self.m1, self.m2, self.m3, self.m4)
        elif self.currentCommand=='winkel':
            self.drone.winkel(self.w1,self.w2,self.w3,self.w4)
        elif self.currentCommand=='autowinkel':
            self.drone.winkel(self.w1,self.w2,self.w3,self.w4)
        elif self.currentCommand=='land':
            self.land()
            
#        print 'self.drone.navdata: ' + str(self.drone.navdata) 


    def takeOff(self):
        print 'ArDrone Take Off'
        self.drone.reset()
#        time.sleep(2)
        self.drone.takeoff()
        self.flying = True
        self.currentCommand=''
    
    def land(self):
        print 'ArDrone Landing...'
        self.flying = False
        self.drone.land()
        self.currentCommand=''
        self.drone.halt()
    
    def command(self,com):
        self.currentCommand=com
        self.fly()
        
        
    def decode_navdata(packet):
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

    def main(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.drone = libardrone.ARDrone()
        self.drone.speed=1

if __name__ == '__main__':
    
    cl = pygame.time.Clock()
    cl.tick(10)
    print 'start'
    
    ac= ArdroneCommander()

    print 'land'
    ac.command('land')
    cl.tick(.5)
    
    print "Shutting down...",
    ac.drone.halt()
    print "Ok."
    
    

