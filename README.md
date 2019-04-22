# Deep Racer Env

This repository contains a simulator for AWS Deep Racer. At this point all of the constants are just my personal guessing. It runs using several dependencies including Shapely, pyOpenGL, pygame, and gym. And currently it must create a window to run. I also provided an implementation of PPO for training along with a jupyter notebook that shows how to use it. To use PPO, you will first need to install [pytorch](https://pytorch.org/get-started/locally/). Good luck!

### Training with PPO

0 epochs

![](gifs/0.gif)

25 epochs

![](gifs/25.gif)

```python
import gym
import gym_deepracer
import time

env = gym.make('deepracer-v0')   # starts as 1000x600
env.update_random_settings({'car_rand_loc':False})
env.resize(128,128) # resize to 128x128 for learning

state = env.reset()

for _ in range(200):
    throttle = 2  # accelerate at 2 m/s^2
    turn = 15     # turn wheels 15 degrees
    action = (throttle, turn)
    state, reward, done, _ = env.step(action)
    time.sleep(1/10) # run at 10fps
env.quit()
```

## Install
#### Some notes on the install process
Before anything else, make sure you have python 3 installed. I recommend getting [anaconda](https://www.anaconda.com/distribution/). Next you need to install everything in the requirements.txt file.
```bash
pip install Shapely gym pygame pyOpenGL numpy imageio scikit-image
```
And if you want to install pytorch, look at [pytorch](https://pytorch.org/get-started/locally/) for instructions specific to your computer.
