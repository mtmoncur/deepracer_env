import os
import time
import math
from copy import copy
from itertools import combinations

import pygame
import numpy as np
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
        s._fps = 10
        s.default_car = Car(187,531-463, fps=s._fps, view_angle=-65)
        s.car = copy(s.default_car)

        #get track shape
        pts_arr = np.load(os.path.join(path,"track_points.npy"))
        s.track_center = LinearRing(pts_arr)
        s.track_cut = Polygon(pts_arr)
        s.track_width = 39
        s.track_shape = s.track_center.buffer(s.track_width)

        # create display
        s.track_img = imread(os.path.join(path,"aws-track2.png"))
        s.resize(width, height)

        #only used for RL, not in human mode
        s.time = 0          # time measured in frames
        s.driving_dist = 0  # true driving distance
        s.track_dist = 0    # only incremented while on track

        def base_reward(params):
            if params['all_wheels_on_track']:
                return 1.0
            else:
                return 0.0
        s._reward_func = base_reward

        s.random_settings = {
            'car_bias':False,
            'car_rand':False,
            'car_rand_loc':True,
            'disp_bias':False,
            'disp_rand':False,
            'track_fixed_noise':True,
            'track_rand_noise':False,
            'track_rand_color':False,
            'track_rand_light':False
        }

    def update_reward_func(s, reward_func):
        s._reward_func = reward_func

    def resize(s, width, height):
        if hasattr(s, 'win'): s.quit()
        pygame.init()
        s.win = pygame.display.set_mode((width, height), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Deep Racer")
        s.display = Display(fr_height=height, fr_width=width, img=s.track_img)

        #initialize display camera
        init_dir = np.rad2deg(s.car.direction)
        s.display.rotate_x(s.car.view_angle)
        s.display.rotate_z_abs(init_dir) #point the camera forward
        s.display.draw()

        s.time = 0

    def step(s, action):
        """Apply action, return new state, reward, done, empty info dict"""
        s.throttle, s.steering_angle = action
        prev_dist = s.get_distance_to_center()
        s.move_car(s.throttle, s.steering_angle)
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
            delta_track_dist = cur_track_point.distance(s.prev_track_point)
            s.track_dist += delta_track_dist
        s.prev_track_point = cur_track_point

        params = s.get_params()
        reward = s._reward_func(params)
        state = s.get_state()
        done = ((not is_display_alive) or (abs(s.get_distance_to_center()) > 80))

        return state, reward, done, {}

    def render(s, mode='human', close=False):
        """Generate image for display. Return the viewer."""
        if (mode=='rgb_array') and hasattr(s, 'camera_view'):
            return s.camera_view
        else:
            return None

    def reset(s):
        """Set everything back and return observation."""
        s.time = 0
        if s.random_settings['car_rand_loc']:
            s.car = s.random_car_loc()
        else:
            s.car = copy(s.default_car)
        s.randomize_track()
        if hasattr(s, 'prev_track_point'):
            delattr(s, 'prev_track_point')
        is_display_alive = s.draw()
        s.camera_view = s.display.read_screen()
        return s.get_state()

    def get_params(s):
        params = {}
        params['all_wheels_on_track'] = s.is_on_track()
        params['x'] = s.car.x/s.car.m_to_px
        params['y'] = s.car.y/s.car.m_to_px
        params['distance_from_center'] = abs(s.get_distance_to_center())/s.car.m_to_px
        params['is_left_of_center'] = (s.get_distance_to_center() < 0)
        params['heading'] = math.degrees(s.car.direction)%360
        params['progress'] = s.get_progress()
        params['steps'] = s.time
        params['speed'] = s.car.v
        params['steering_angle'] = s.steering_angle
        params['track_width'] = s.track_width / s.car.m_to_px
        return params

    def get_progress(s):
        return 100 * s.track_dist / s.track_center.length

    def get_state(s):
        image = s.camera_view.astype(np.float32)/255
        env_state = np.array([s.time/100, s.car.v], dtype=np.float32) # include variables only known to the environment
        other_state = np.zeros(1, dtype=np.float32) # should include gyroscope and accelerometer here
        return image, env_state, other_state

    def get_angle(s, x1, y1, x2, y2):
        center_point_x = 800/2 # random point inside of track
        center_point_y = 531/2
        angle1 = np.arctan2(x1-center_point_x, y1-center_point_y)
        angle2 = np.arctan2(x2-center_point_x, y2-center_point_y)
        return angle2 - angle1

    def is_on_track(s):
        pos = Point((s.car.x,s.car.y))
        return s.track_shape.contains(pos)

    def get_distance_to_center(s):
        pos = Point((s.car.x,s.car.y))
        sign = -1 if s.track_cut.contains(pos) else 1
        return sign*s.track_center.distance(pos)

    def update_random_settings(s, new_settings):
        """To make an agent trained in the virtual environments
        capable of handling real scenarios, we add randomness.

        Parameters
        ----------
            new_settings : dict
                Dictionary to update the random settings. Must only contain
                keys of predifined settings.

        Possible settings:
            car_bias - add permanent to car inputs for entire rollout
            car_rand - add random noise to car inputs for each step
            car_rand_loc - start the car at a random location on the track
            disp_bias - not implemented
            disp_rand - not implemented
            track_fixed_noise - add poisson noise to track colors for each rollout
            track_rand_color - randomize color scheme of track for each rollout
            track_rand_light - randomize track brightness for each rollout

        Examples
        --------
        >>> import gym
        >>> import gym_deepracer
        >>> env = gym.make('deepracer-v0')
        >>> env.update_random_settings({'car_rand_loc':False})
        """
        prev_len = len(s.random_settings)
        s.random_settings.update(new_settings)
        assert len(s.random_settings) == prev_len, "Unknown key in random_settings dictionary."
        for k,v in s.random_settings.items():
            if k.startswith('car_'):
                s.car.random_settings[k] = v
            elif k.startswith('disp_'):
                s.display.random_settings[k] = v
        s.car.update_random_settings()
        s.display.update_random_settings()

    def random_car_loc(s):
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
        if np.random.rand() < 0.1: return
        norm = np.linalg.norm
        img_r = s.track_img.copy()
        if s.random_settings['track_rand_color'] and np.random.rand() < 0.5:
            # 50% chance of new color scheme, 50% chance of original color scheme
            colors = np.array([[49,169,141],[47,61,69],[255,255,255],[238, 163,  85]])
            new_colors = s.random_colors(4)
            img_r[norm(s.track_img.astype(np.float32) - colors[0], axis=2) < 80] = new_colors[0]
            img_r[norm(s.track_img.astype(np.float32) - colors[1], axis=2) < 80] = new_colors[1]/3 #make track dark
            img_r[norm(s.track_img.astype(np.float32) - colors[2], axis=2) < 80] = new_colors[2]
            img_r[norm(s.track_img.astype(np.float32) - colors[3], axis=2) < 80] = new_colors[3]

        if s.random_settings['track_fixed_noise']:
            img_r = (255*skimage.util.random_noise(img_r, mode='poisson')).astype(np.uint8)
        if s.random_settings['track_rand_light']:
            brightness = np.random.beta(5, 1)
            img_r = (brightness*img_r).astype(np.uint8)
        s.display.new_track(img_r)

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
        for _ in range(400):
            pygame.time.delay(1000//s._fps-2)
            count += 1
            s.move_car_with_keys()
            run = s.draw()

            if ((count+1)%20==0):
                print(f"frameRate: {20/(time.clock() - start)}")
                start = time.clock()
        s.quit()

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

    def move_car(s, throttle, turn):
        """RL mode to move car."""
        s.car.throttle(throttle)
        s.car.turn(turn)
        s.car.update()

    def move_car_with_keys(s):
        """Used in interactive mode to get key pushes and move car."""
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            s.car.turn(20)
        if keys[pygame.K_RIGHT]:
            s.car.turn(-20)
        if keys[pygame.K_UP]:
            s.car.throttle(5)
        if keys[pygame.K_DOWN]:
            s.car.throttle(-5)
        s.car.update()

class DeepRacerEnvDiscrete(DeepRacerEnv):
    metadata = {'render.modes':['human']}
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(9)
        self.turn_options = {
            0: 20,
            1: 15,
            2: 10,
            3: 5,
            4: 0,
            5: -5,
            6: -10,
            7: -15,
            8: -20}

        self.throttle = 5

    def step(self, action):
        """
        Apply action, return new state, reward, done, empty info dict
        Parameters:
            action (int 0-6) - the integer of the action to take

        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.steering_angle = self.turn_options[action]
        return super().step((self.throttle, self.steering_angle))

if __name__ == "__main__":
    main()
