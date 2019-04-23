import numpy as np

class Car:
    """
    This Class models the movements and relevant information of the DeepRacer car.
    You are not meant to interact directly with objects of this type, but rather it is
    used by the DeepRacerEnv
    """
    def __init__(s, x, y, view_angle, fps, direction=0):
        s.x = x # px
        s.y = y # px
        s.view_angle = view_angle # radians
        s.direction = direction # radians
        s.v = 0 # m
        s.turn_angle = 0 # radians
        s.dx = 0 # px
        s.dy = 0 # px
        s.throttle_set = False
        s.path = np.array([[x,y]]*6, dtype=np.float32)  # contains 4 previous points

        # constants
        s.delta_t = 1/fps # s
        s.m_to_px = 800/7.92 # px/m  (800px = 7.92m)
        s.set_max_values(v=1.5, a=5)
        s.min_drag = 0.4 # m/s^2
        s._length = 0.25 # m

        # setup randomness
        s.random_settings = {
            'car_rand':False,
            'car_bias':False
        }
        s.update_random_settings()

    def set_max_values(s, v, a):
        s.max_v = v # m/s (2m/s ~ 4.5 mph)
        s.max_a = a # m/s^2
        s.drag_coef = s.max_a/(s.max_v**2) # 1/m (drag that enforces max_v)

    def update_random_settings(s):
        if s.random_settings['car_bias']:
            s.v_bias = np.clip(np.random.rand()/20, -0.05, 0.05)
            s.t_angle_bias = np.clip(np.random.rand(), -1, 1)
        else:
            s.v_bias = 0
            s.t_angle_bias = 0

        if s.random_settings['car_rand']:
            s.v_stddev = 1/20      # m/s^2
            s.t_angle_stddev = 1/8 # degrees
        else:
            s.v_stddev = 0
            s.t_angle_stddev = 0

    def throttle(s, throttle):
        """
        Input throttle in terms of acceleration (m/s^2).
        """
        throttle = _rand(throttle, s.v_bias, s.v_stddev)
        throttle = max(min(throttle,s.max_a), -s.max_a)
        drag = np.sign(s.v) * (s.drag_coef*(s.v)**2 + s.min_drag)
        if (abs(drag*s.delta_t) > abs(s.v)):
            drag = s.v/s.delta_t
        dv = s.delta_t * (throttle - drag)
        s.v += dv
        s.throttle_set = True

    def turn(s, turn_angle):
        s.turn_angle += np.deg2rad(_rand(turn_angle, s.t_angle_bias, s.t_angle_stddev))

    def update(s):
        if not s.throttle_set: s.throttle(0)
        if abs(s.turn_angle) < 0.02:
            s.dx = np.cos(s.direction) * s.v * s.delta_t
            s.dy = np.sin(s.direction) * s.v * s.delta_t
        else:
            # use circle steering math (lookup ackerman steering)
            r = s._length/np.sin(s.turn_angle)
            tau = s.v*s.delta_t/r
            rel_dx = r*(np.sin(tau+s.turn_angle)) - s._length
            rel_dy = -r*(np.cos(tau+s.turn_angle) - np.cos(s.turn_angle))

            # rotate
            s.dx = rel_dx*np.cos(s.direction) - rel_dy*np.sin(s.direction)
            s.dy = rel_dx*np.sin(s.direction) + rel_dy*np.cos(s.direction)
            s.direction += tau

        #convert to px
        s.dx *= s.m_to_px
        s.dy *= s.m_to_px
        s.x += s.dx
        s.y += s.dy

        # reset action values
        s.turn_angle = 0
        s.throttle_set = False

        # update path
        s.path = np.roll(s.path, 1)
        s.path[0] = np.array([s.x,s.y])

    def get_acceleration(s):
        f = s.path
        h = s.delta_t
        a = (-7*f[5]+208*f[2]-330*f[1]+112*f[0]+17*f[3])/(240*1.0*h**2)
        return a

    def get_gyro(self):
        pass

def _rand(val, bias, std_dev):
    return np.random.normal(val + bias, std_dev)
