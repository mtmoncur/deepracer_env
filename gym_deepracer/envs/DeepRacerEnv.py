import os
import time
import pygame, OpenGL
import numpy as np
import matplotlib.pyplot as plt
import gym

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from Display import Display

scl = 9
width, height = 1000, 600

# os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['DISPLAY'] = ':0.0'

def main():
    env = DeepRacerEnv()
    env.test()
    # env.play()

class Car:
    """
    This Class models the movements and relevant information of the DeepRacer car.
    You are not meant to interact directly with objects of this type, but rather it is
    used by the DeepRacerEnv
    """
    def __init__(s, x, y, view_angle):
        s.x = x
        s.y = y
        s.view_angle = view_angle
        s.v = 0
        s.max_v = 70
        s.drag = 3
        s.direction = 0
        s.turn_angle = 0
        
        s.ddirection = 0
        s.dx = 0
        s.dy = 0

    def throttle(s, throttle):
        s.v += throttle
        s.v = max(min(s.v,s.max_v), -s.max_v)

    def turn(s, turn_angle):
        s.turn_angle += np.deg2rad(turn_angle)

    def update(s):
        s.ddirection = s.turn_angle * s.v/s.max_v
        s.direction += s.ddirection

        s.dx = np.cos(s.direction) * s.v
        s.dy = np.sin(s.direction) * s.v
        s.x += s.dx
        s.y += s.dy

        s.v = min(s.v+s.drag,0) + max(s.v-s.drag, 0)
        s.turn_angle = 0

class DeepRacerEnv(gym.Env):
    metadata = {'render.modes':['human']}
    
    def __init__(s):
        pygame.init()
        s.win = pygame.display.set_mode((width, height), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Deep Racer")
        s.display = Display(fr_height=height, fr_width=width, scl=scl, filename='aws-track.png')
        s.car = Car(0,0, view_angle=-65)
        #initialize display camera
        s.display.rotate_x(s.car.view_angle)
        s.display.translate(0,-0.83) #move the point of rotation to bottom of screen
        
    def step(s, action):
        """Apply action, return new obs, reward, done, probabilities"""
        throttle, turn = action
        s.move_car(throttle, turn)
        is_display_alive = s.draw()
        s.camera_view = s.display.read_screen()
        
        reward = 0
        state = [s.camera_view]
        done = is_display_alive
        
        return state, reward, done, {}
        
    
    def reset(s):
        """Set everything back and return obs."""
        s.car = Car(0,0, view_angle=-65)
        is_display_alive = s.draw()
        s.camera_view = s.display.read_screen()
        
        state = [s.camera_view]
        
        return state
        
    
    def render(s, mode='human', close=False):
        """Generate image for display. Return the viewer."""
        if (mode=='rgb_array') and hasattr(s, 'camera_view'):
            return s.camera_view
        else:
            return None

    def test(s):
        """Quickly run the car in a circle for 1000 steps for testing purposes."""
        run = True
        count = 0
        start = time.clock()
        # while run:
        for _ in range(1000):
            pygame.time.delay(0)
            count += 1
            s.move_car(throttle=4,turn=3)
            run = s.draw()
            img = s.display.read_screen()
            if np.random.random() < 0.01:
                print(img.shape)
                plt.imshow(img)
                plt.show()

            if ((count+1)%100==0):
                print(f"frameRate: {100/(time.clock() - start)}")
                start = time.clock()
        s.quit()

    def play(s):
        """Interactive-mode. You get to drive the car."""
        run = True
        count = 0
        start = time.clock()
        while run:
        # for _ in range(100000):
            pygame.time.delay(20)
            count += 1
            s.move_car_with_keys()
            run = s.draw()

            if ((count+1)%100==0):
                print(f"frameRate: {100/(time.clock() - start)}")
                start = time.clock()
        s.quit()

    def quit(s):
        pygame.quit()

    def draw(s):
        """Unlike most Environments, this one must render the scene to create the observations."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
        # Explanation
        # When created in __init__, the camera is raised and tilted so we are not looking straight at the ground
        # First move the image to simulate motion, (we leave the camera at the origin for rotation reasons)
        # Second rotate the camera to simulate turning
        s.display.translate_img(s.car.dy/2500, -s.car.dx/2500) # 2500 is a large number that works well
        #s.display.move_img_to(s.car.y/2500, -s.car.x/2500) # 2500 is a large number that works well
        s.display.rotate_z(np.rad2deg(-s.car.ddirection))
        s.display.draw()
        return True

    def move_car(s, throttle, turn):
        """RL mode to move car."""
        assert abs(throttle) < 30 #arbitrary
        assert abs(turn) < 15  #arbitrary
        s.car.throttle(throttle)
        s.car.turn(turn)
        s.car.update()

    def move_car_with_keys(s):
        """Used in interactive mode to get key pushes and move car."""
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            s.car.turn(3)
        if keys[pygame.K_RIGHT]:
            s.car.turn(-3)
        if keys[pygame.K_UP]:
            s.car.throttle(7)
        if keys[pygame.K_DOWN]:
            s.car.throttle(-7)
        s.car.update()

if __name__ == "__main__":
    main()
