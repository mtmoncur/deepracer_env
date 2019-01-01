# Deep Racer Env

This repository contains a simulator for AWS Deep Racer. At this point all of the just my personal guessing. It runs using pyOpenGL, pygame, and gym, and it currently must create a window to run.

### Training with PPO

Random Initialization (0 epochs)

![](gifs/0.gifs)

15 epochs

![](gifs/15.gifs)


```import gym
import gym_deepracer
import time

env = gym.make('deepracer-v0')   # starts as 1000x600
env.resize(128,128,random=False) # resize to 128x128 for learning

camera_view = env.reset()
for _ in range(1000):
    throttle = 3
    turn = 1 # drive in a circle
    action = (throttle, turn)
    camera_view, reward, done, _ = env.step(action)
    time.sleep(1/30) # run at 30fps
```
