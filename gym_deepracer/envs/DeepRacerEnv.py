import os
import time
import pygame, OpenGL
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing,Point
from shapely.ops import nearest_points
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
        s.max_v = 6
        s.drag = 1
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
#         s.display.translate(0,-0.83) #move the point of rotation to bottom of screen
#         s.display.translate(0,-0.83*height) #move the point of rotation to bottom of screen

        #get track shape
        pts_arr = np.load("track_points.npy")
        s.track_center = LinearRing(pts_arr)
        s.track_shape = s.track_center.buffer(39) # track is about 39 pixels wide

        #only used for RL, not in human mode
        s.time = 0 # time measured in frames
        s.driving_dist = 0  # true driving distance
        s.track_dist = 0    # only incremented while on track
        s.prev_track_point = nearest_points(s.track_center, Point(-s.car.y, -s.car.x))[0]

    def step(s, action):
        """Apply action, return new state, reward, done, empty info dict"""
        throttle, turn = action
        s.move_car(throttle, turn)
        is_display_alive = s.draw()
        s.camera_view = s.display.read_screen()

        # increment measurements
        s.time += 1
        s.driving_dist += np.sqrt(s.car.dx**2+s.car.dy**2)
        cur_point = Point(-s.car.y, -s.car.x)
        cur_track_point = nearest_points(s.track_center, cur_point)[0]
        if s.is_on_track():
            s.track_dist += cur_track_point.distance(s.prev_track_point)
        s.prev_track_point = cur_track_point

        # finalize other values
        reward = 0 #implement your own reward system here
        state = [s.camera_view] # true system includes camera, gyroscope,and accelerometer
        done = not is_display_alive # implement your own logic on when to be done

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
            s.move_car(throttle=1,turn=3)
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
        s.display.translate_img(s.car.dy, -s.car.dx)
        #s.display.move_img_to(s.car.y/2500, -s.car.x/2500) # 2500 is a large number that works well
        s.display.rotate_z(np.rad2deg(-s.car.ddirection))
        s.display.draw()
        return True

    def is_on_track(s):
        # negatives since we move the image instead of the camera
        pos = Point((-s.car.y,-s.car.x))
        return s.track_shape.contains(pos)

    def distance_to_centerline(s):
        # negatives since we move the image instead of the camera
        pos = Point((-s.car.y,-s.car.x))
        return s.track_center.distance(pos)

    def move_car(s, throttle, turn):
        """RL mode to move car."""
        assert abs(throttle) < 20 #arbitrary
        assert abs(turn) < 15  #arbitrary
        s.car.throttle(throttle)
        s.car.turn(turn)
        s.car.update()

    def move_car_with_keys(s):
        """Used in interactive mode to get key pushes and move car."""
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            s.car.turn(1.5)
        if keys[pygame.K_RIGHT]:
            s.car.turn(-1.5)
        if keys[pygame.K_UP]:
            s.car.throttle(3)
        if keys[pygame.K_DOWN]:
            s.car.throttle(-3)
        s.car.update()

if __name__ == "__main__":
    main()
