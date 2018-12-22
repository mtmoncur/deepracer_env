import os
import time
import pygame, OpenGL
import numpy as np
import matplotlib.pyplot as plt

from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from Display import Display

# from scipy.misc import imread
# from scipy.interpolate import LinearNDInterpolator
scl = 9
width, height = 1000, 600

# os.environ["SDL_VIDEODRIVER"] = "dummy"

def main():
    env = DeepRacerEnv()
    env.test()
    # env.play()
    # env.make(mode='human')

class Car:
    def __init__(s, x, y, view_angle):
        s.x = x
        s.y = y
        s.view_angle = view_angle
        s.v = 0
        s.max_v = 70
        s.drag = 3
        s.direction = 0
        s.turn_angle = 0

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

class DeepRacerEnv:
    def __init__(s):
        pygame.init()
        s.win = pygame.display.set_mode((width, height), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Deep Racer")
        s.display = Display(fr_height=height, fr_width=width, scl=scl, filename='aws-track.png')
        s.car = Car(0,0, view_angle=-65)
        #initialize display camera
        s.display.rotate_x(s.car.view_angle)
        s.display.translate(0,-0.83) #move the point of rotation to bottom

    def test(s):
        run = True
        count = 0
        start = time.clock()
        # while run:
        for _ in range(1000):
            pygame.time.delay(0)
            count += 1
            s.car.throttle(4)
            s.car.turn(3)
            s.car.update()
            run = s.draw()
            img = s.display.read_screen()
            if np.random.random() < 0.01:
                print(img.shape)
                plt.imshow(img[::-1])
                plt.show()

            if ((count+1)%100==0):
                print(f"frameRate: {100/(time.clock() - start)}")
                start = time.clock()
        s.quit()

    def play(s):
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        s.display.translate_img(s.car.dy/2500, -s.car.dx/2500) #large number that works
        s.display.rotate_z(np.rad2deg(-s.car.ddirection))
        s.display.draw()
        return True

    def move_car(s, throttle, turn):
        assert abs(throttle) < 30 #arbitrary
        assert abs(turn) < 15  #arbitrary
        s.car.throttle(throttle)
        s.car.turn(turn)
        s.car.update()

    def move_car_with_keys(s):
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
