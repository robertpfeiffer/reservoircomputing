import pygame

import libardrone
import time


#pygame.init()
#W, H = 320, 240
#screen = pygame.display.set_mode((W, H))
#drone = libardrone.ARDrone()
#clock = pygame.time.Clock()
#
#drone.land()
#drone.halt()


drone = libardrone.ARDrone()

drone.land()
    
drone.halt()
#
#drone.directMotor(1, 1, 1, 1)
#time.sleep(10)
#drone.directMotor(0, 0, 0, 0)


