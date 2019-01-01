import os
import time
from copy import copy
import pygame, OpenGL
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing,Point
from shapely.ops import nearest_points
import gym

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from gym_deepracer.envs.Display import Display

path = os.path.dirname(os.path.abspath(__file__))

# os.environ['DISPLAY'] = ':0.0'
np.random.seed(int(time.time()))
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
    def __init__(s, x, y, view_angle, direction=0):
        s.x = x
        s.y = y
        s.view_angle = view_angle
        s.v = 0
        s.max_v = 5
        s.drag = 0.6
        s.direction = direction
        s.turn_angle = 0
        
        s.ddirection = s.direction
        s.dx = x
        s.dy = y

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
    
    def __init__(s, width=1000, height=600, scl = 1):
        super().__init__()
#         s.mirror = False if np.random.rand() < 0.5 else True
#         if s.mirror:
#             s.default_car = Car(187,453, view_angle=-65)
#         else:
        s.default_car = Car(187,531-453, view_angle=-65)
            
        #get track shape
        pts_arr = np.load(os.path.join(path,"track_points.npy"))
#         if s.mirror:
#             pts_arr[:,1] = 531 - pts_arr[:,1]
        s.track_center = LinearRing(pts_arr)
        s.track_shape = s.track_center.buffer(39) # track is about 39 pixels wide
        
        # create display
        s.resize(width, height, scl)

        #only used for RL, not in human mode
        s.time = 0 # time measured in frames
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
#                     print("rand", x, y, direction)
#                     if np.sign(s.get_angle(pt2.x,pt2.y,pt1.x,pt1.y)) < 0:
                    if np.random.rand() < 0.5:
                        direction += np.pi
                    new_car = Car(x, y, default_angle, direction)
                    return new_car
                    
    def resize(s, width, height, scl=1, random=True):
        if hasattr(s, 'win'): s.quit()
        pygame.init()
        s.win = pygame.display.set_mode((width, height), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Deep Racer")
        s.display = Display(fr_height=height, fr_width=width, scl=scl, filename=os.path.join(path,'aws-track.png'))
        if random:
            s.car = s.random_car()
        else:
            s.car = copy(s.default_car)

        #initialize display camera
        init_dir = np.rad2deg(s.car.direction)
        s.display.rotate_x(s.car.view_angle)
        s.display.translate(0,-0.15*height)
        s.display.rotate_z_abs(init_dir) #point the camera forward
        s.display.draw()
        
    def get_angle(s, x1, y1, x2, y2):
        center_point_x = 800/2 # random point insided of track
        center_point_y = 531/2
        angle1 = np.arctan2(x1-center_point_x, y1-center_point_y)
        angle2 = np.arctan2(x2-center_point_x, y2-center_point_y)
        return angle2 - angle1

    def step(s, action):
        """Apply action, return new state, reward, done, empty info dict"""
        if hasattr(action, '__getitem__'):
            throttle, turn = action
        else:
            turn = action - 3
            throttle = 3
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
            angle = s.get_angle(cur_track_point.x, cur_track_point.y,
                                s.prev_track_point.x, s.prev_track_point.y)
            if (True):#angle > 0):
                delta_track_dist = cur_track_point.distance(s.prev_track_point)
                s.track_dist += delta_track_dist
        s.prev_track_point = cur_track_point

        # finalize other values
        reward = delta_track_dist#1.0 if s.is_on_track() else 0.0 #1.0 #implement your own reward system here
        state = [s.camera_view][0] # true system includes camera, gyroscope,and accelerometer
        done = ((not is_display_alive) or (s.distance_to_centerline() > 50)) #or not s.is_on_track() # implement your own logic on when to be done

        return state, reward, done, {}

    def reset(s, random=True):
        """Set everything back and return obs."""
#         prev_dir = 
        if random:
            s.car = s.random_car()
        else:
            s.car = copy(s.default_car)
        if hasattr(s, 'prev_track_point'):
            delattr(s, 'prev_track_point')
        is_display_alive = s.draw()
        s.camera_view = s.display.read_screen()

        state = [s.camera_view][0]

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
            s.move_car(throttle=1,turn=3)
            run = s.draw()
            if not run: break
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
#         s.display.translate_img(s.car.dx, s.car.dy)
        s.display.move_img_to(s.car.x, s.car.y)
#         s.display.rotate_z(np.rad2deg(-s.car.ddirection))
        s.display.rotate_z_abs(np.rad2deg(-s.car.direction))
        s.display.draw()
        return True

    def is_on_track(s):
        # negatives since we move the image instead of the camera
#         pos = Point((-s.car.y,-s.car.x))
        pos = Point((s.car.x,s.car.y))
        return s.track_shape.contains(pos)

    def distance_to_centerline(s):
        # negatives since we move the image instead of the camera
        pos = Point((s.car.x,s.car.y))
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
