import os
import time
from copy import copy
from itertools import combinations

import pygame
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,LinearRing,Point
from shapely.ops import nearest_points
from imageio import imread
import skimage
import gym
from gym import spaces

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from gym_deepracer.envs.Display import Display
from gym_deepracer.envs.Car import Car

path = os.path.dirname(os.path.abspath(__file__))

np.random.seed(int(time.time()))
def main():
    env = DeepRacerEnv()
    env.test()
    # env.play()

class DeepRacerEnv(gym.Env):
    metadata = {'render.modes':['human']}
    
    def __init__(s, width=1000, height=600):
        super().__init__()
        s.random = False
        s._fps = 30
        s.default_car = Car(187,531-463, fps=s._fps, view_angle=-65)

        #get track shape
        pts_arr = np.load(os.path.join(path,"track_points.npy"))
        s.track_center = LinearRing(pts_arr)
        s.track_cut = Polygon(pts_arr)
        s.track_shape = s.track_center.buffer(39) # track is about 39 pixels wide

        # create display
        s.track_img = imread(os.path.join(path,"aws-track2.png"))
        s.resize(width, height)

        #only used for RL, not in human mode
        s.time = 0          # time measured in frames
        s.driving_dist = 0  # true driving distance
        s.track_dist = 0    # only incremented while on track

    def random_car(s):
        while True:
            x = np.random.uniform(0,s.display.width)
            y = np.random.uniform(0,s.display.height)
            pt = Point(x,y)
            if s.track_center.distance(pt) < 20:
                default_angle = s.default_car.view_angle
                #now find the direction
                pt1 = nearest_points(s.track_center, pt)[0]
                pt2 = nearest_points(s.track_center, Point(x+0.03, y+0.03))[0]
                direction = np.arctan2(pt2.y-pt1.y, pt2.x-pt1.x)
                # commented out to allow forwards and backwards
                # if np.sign(s.get_angle(pt2.x,pt2.y,pt1.x,pt1.y)) < 0:
                if np.random.rand() < 0.5:
                    direction += np.pi
                new_car = Car(x, y, default_angle, s._fps, direction)
                return new_car

    def random_colors(s, size):
        while True:
            new_colors = np.random.randint(0,256,(size,3))
            color_list = list(new_colors)
            for c1,c2 in combinations(color_list, r=2):
                if np.linalg.norm(c1.astype(np.float32)-c2) < 120:
                    break
            else:
                return new_colors

    def randomize_track(s):
        norm = np.linalg.norm
        if np.random.rand() < 0.1:
            colors = np.array([[49,169,141],[47,61,69],[255,255,255],[238, 163,  85]])
            img_r = s.track_img.copy()
            new_colors = s.random_colors(4)
            img_r[norm(s.track_img.astype(np.float32) - colors[0], axis=2) < 80] = new_colors[0]
            img_r[norm(s.track_img.astype(np.float32) - colors[1], axis=2) < 80] = new_colors[1]
            img_r[norm(s.track_img.astype(np.float32) - colors[2], axis=2) < 80] = new_colors[2]
            img_r[norm(s.track_img.astype(np.float32) - colors[3], axis=2) < 80] = new_colors[3]

            new_track_img = (255*skimage.util.random_noise(img_r, mode='poisson')).astype(np.uint8)
            s.display.new_track(new_track_img)
        elif np.random.rand() < 0.2:
            s.display.new_track(s.track_img)
        else:
            pass

    def set_random(s, mode):
        s.random = mode

    def resize(s, width, height, img=None):
        if hasattr(s, 'win'): s.quit()
        pygame.init()
        s.win = pygame.display.set_mode((width, height), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Deep Racer")
        if img is not None:
            s.display = Display(fr_height=height, fr_width=width, img=img)
        else:
            s.display = Display(fr_height=height, fr_width=width, img=s.track_img)
        if s.random:
            s.car = s.random_car()
        else:
            s.car = copy(s.default_car)

        #initialize display camera
        init_dir = np.rad2deg(s.car.direction)
        s.display.rotate_x(s.car.view_angle)
        s.display.rotate_z_abs(init_dir) #point the camera forward
        s.display.draw()

        s.time = 0

    def get_angle(s, x1, y1, x2, y2):
        center_point_x = 800/2 # random point insided of track
        center_point_y = 531/2
        angle1 = np.arctan2(x1-center_point_x, y1-center_point_y)
        angle2 = np.arctan2(x2-center_point_x, y2-center_point_y)
        return angle2 - angle1

    def step(s, action):
        """Apply action, return new state, reward, done, empty info dict"""
        throttle, turn = action
        prev_dist = s.distance_to_centerline()
        s.move_car(throttle, turn)
        is_display_alive = s.draw()
        s.camera_view = s.display.read_screen()

        # increment measurements
        s.time += 1
        s.driving_dist += np.sqrt(s.car.dx**2+s.car.dy**2)
        cur_point = Point(s.car.x, s.car.y)
        if not hasattr(s, 'prev_track_point'):
            s.prev_track_point = nearest_points(s.track_center, Point(s.car.x, s.car.y))[0]
        cur_track_point = nearest_points(s.track_center, cur_point)[0]
        delta_track_dist = 0.0
        if s.is_on_track():
            #angle = s.get_angle(cur_track_point.x, cur_track_point.y,
            #                    s.prev_track_point.x, s.prev_track_point.y)
            #if(angle > 0): # prevent reward for backwards movement
            if (True):
                delta_track_dist = cur_track_point.distance(s.prev_track_point)
                s.track_dist += delta_track_dist
        s.prev_track_point = cur_track_point

        # finalize other values
        #reward = delta_track_dist/s.car.max_v
        #reward = (abs(prev_dist) - abs(s.distance_to_centerline()))/70
        reward = 1.0 if s.is_on_track() else 0.0
        state = [s.camera_view.astype(np.float32)/255, np.array([s.time/100])] # true system includes camera, gyroscope,and accelerometer
        done = ((not is_display_alive) or (abs(s.distance_to_centerline()) > 80)) #implement your own logic on when to be done

        return state, reward, done, {}

    def reset(s):
        """Set everything back and return observation."""
        s.time = 0
        if s.random:
            s.car = s.random_car()
            #s.randomize_track()
        else:
            s.car = copy(s.default_car)
        if hasattr(s, 'prev_track_point'):
            delattr(s, 'prev_track_point')
        is_display_alive = s.draw()
        s.camera_view = s.display.read_screen()
        state = [s.camera_view.astype(np.float32)/255, np.array([s.time/100])]

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
        for _ in range(1000):
            pygame.time.delay(0)
            count += 1
            s.move_car(throttle=1,turn=20)
            run = s.draw()
            if not run: break

            if ((count+1)%100==0):
                print(f"frameRate: {100/(time.clock() - start)}")
                start = time.clock()
        s.quit()

    def play(s):
        """Interactive-mode. You get to drive the car."""
        run = True
        count = 0
        start = time.clock()
        v = []
#         while run:
        for _ in range(400):
            v.append(s.car.v)
            pygame.time.delay(1000//s._fps-5)
            count += 1
            s.move_car_with_keys()
            run = s.draw()

            if ((count+1)%100==0):
                if s.is_on_track(): print("===On track===")
                print(f"frameRate: {100/(time.clock() - start)}")
                start = time.clock()
        s.quit()
        plt.plot(v)
        plt.show()

    def quit(s):
        pygame.quit()

    def draw(s):
        """Unlike most Environments, this one must render the scene to create the observations."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Full Drawing Explanation
        # When created in __init__, the camera is raised and tilted so we are not looking straight at the ground
        # First move the image to simulate motion, (we leave the camera at the origin for rotation reasons)
        # Second rotate the camera to simulate turning
        s.display.move_img_to(s.car.x, s.car.y)
        s.display.rotate_z_abs(np.rad2deg(-s.car.direction))
        s.display.draw()
        return True

    def is_on_track(s):
        pos = Point((s.car.x,s.car.y))
        return s.track_shape.contains(pos)

    def distance_to_centerline(s):
        pos = Point((s.car.x,s.car.y))
        sign = -1 if s.track_cut.contains(pos) else 1
        return sign*s.track_center.distance(pos)

    def move_car(s, throttle, turn):
        """RL mode to move car."""
        s.car.throttle(throttle)
        s.car.turn(turn)
        s.car.update()

    def move_car_with_keys(s):
        """Used in interactive mode to get key pushes and move car."""
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            s.car.turn(16)
        if keys[pygame.K_RIGHT]:
            s.car.turn(-16)
        if keys[pygame.K_UP]:
            s.car.throttle(5)
        if keys[pygame.K_DOWN]:
            s.car.throttle(-5)
        s.car.update()

class DeepRacerEnvDiscrete(DeepRacerEnv):
    metadata = {'render.modes':['human']}
    def __init__(self, width=1000, height=600):
        super().__init__()

        self.action_space = spaces.Discrete(6)
        self.turn_options = {
            0: 20,
            1: 13,
            2: 5,
            3: 0,
            4: -5,
            5: -13,
            6: -20}

        self.throttle = 5

    def step(self, action):
        """
        Apply action, return new state, reward, done, empty info dict
        Parameters:
            action (int 0-6) - the integer of the action to take

        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        turn = self.turn_options[action]
        return super().step((self.throttle, turn))

if __name__ == "__main__":
    main()
