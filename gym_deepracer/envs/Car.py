import numpy as np

class Car:
    """
    This Class models the movements and relevant information of the DeepRacer car.
    You are not meant to interact directly with objects of this type, but rather it is
    used by the DeepRacerEnv
    """
    def __init__(s, x, y, view_angle, direction=0, random=True, biased=False):
        s.x = x
        s.y = y
        s.view_angle = view_angle
        s.v = 0
        s.max_v = 5
        s.drag = 0.6
        s.direction = direction
        s.turn_angle = 0

        s.dx = 0
        s.dy = 0
        
        if biased:
            s.v_bias = np.clip(np.random.rand()/10, -0.5, 0.5)
            s.t_angle_bias = np.clip(np.random.rand(), -1.5, 1.5)
        else:
            s.v_bias = 0
            s.t_angle_bias = 0

        if random:
            s.v_stddev = 1/8     # track pixel, not screen pixel
            s.t_angle_stddev = 1/2 # degrees
        else:
            s.v_stddev = 0
            s.t_angle_stddev = 0

    def throttle(s, throttle):
        s.v += _rand(throttle, s.v_bias, s.v_stddev)
        s.v = max(min(s.v,s.max_v), -s.max_v)

    def turn(s, turn_angle):
        s.turn_angle += np.deg2rad(_rand(turn_angle, s.t_angle_bias, s.t_angle_stddev))

    def update(s):
        s.direction += s.turn_angle * s.v/s.max_v
        s.dx = np.cos(s.direction) * s.v
        s.dy = np.sin(s.direction) * s.v
        s.x += s.dx
        s.y += s.dy

        s.v = min(s.v+s.drag,0) + max(s.v-s.drag, 0)
        s.turn_angle = 0

def _rand(val, bias, std_dev):
    return np.random.normal(val + bias, std_dev)