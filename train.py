import os
import pickle
import time
from threading import Thread

from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

import gym_deepracer
from ppo import MLP, Actor, ConvNetwork, ppo

abs_path = os.path.dirname(os.path.abspath(__file__))
print(abs_path)
models_dir = os.path.join(abs_path, 'models')

def get_env():
    # choose to make the discrete action space environment
    env = gym.make('deepracerDiscrete-v0')
    env.update_random_settings({'car_rand_loc':True,
                                'track_fixed_noise':True,
                                'track_rand_color':False,
                                'track_rand_light':True
                               })
    env.resize(128,128) # resize to 128x128 for learning
    return env

class ModelBuilder:
    def __init__(self):
        self._env = get_env()

        self._name_assigned = False
        self._reward_assignd = False
        self._model_built = False
        self._training = False
        self._done_training = False

        required_files = ['gifs','latest_run.csv','weights.torch']
        required_files = set(required_files)
        self._required_files = required_files

    def set_reward(self, reward):
        self._reward_assigned = True
        self._reward = reward
        self._env.update_reward_func(reward)

    def is_used(self, name):
        path = os.path.join(models_dir, name)
        if os.path.isdir(path):
            if self._required_files.issubset(set(os.listdir(path))):
                return True
        return False

    def get_used_names(self):
        ret = ['']
        for name in os.listdir(models_dir):
            if self.is_used(name):
                ret.append(name)
        return ret

    def run_builder_tool(self):
        display(Markdown("## Choose Model Name"))
        @widgets.interact_manual
        def name_model(name=''):
            cur_model_dir = os.path.join(models_dir, name)
            dir_exists = os.path.isdir(cur_model_dir)
            if ('' != name) and (not self.is_used(name)):
                self._name_assigned = True
                if not dir_exists:
                    os.mkdir(cur_model_dir)
                self._model_name = name
                print("{} is a valid unique model name.".format(name))
            else:
                print("Enter a valid new name. Existing model names: {}".format(
                    '\n'.join(self.get_used_names())))

        display(Markdown("## Build Initial Model\n ### Leave empty for random initialization."))
        @widgets.interact_manual(clone_from=self.get_used_names())
        def build(clone_from=''):
            if not self._reward_assigned:
                print("Must create reward function before running.")
                return
            if not self._name_assigned:
                print("Must choose model name before running.")
                return
            latent_size = 100
            hidden_size = 100
            layers = 4
            action_size = self._env.action_space.n

            self._conv_net = ConvNetwork(latent_size)
            self._value = MLP(latent_size, hidden_size, layers, action_size)
            self._policy = Actor(MLP(latent_size, hidden_size, layers, action_size))
            self._model_built = True
            if clone_from == '':
                print("New model initialized with random parameters.")
            else:
                self._load_weights(clone_from)
                print("New model cloned from {}.".format(clone_from))

        display(Markdown("## Choose Hyperparameters/Train Model"))
        @widgets.interact_manual
        def run():
            if not self._model_built:
                print("Must build the model first.")
                return
            if self._done_training:
                print("This model has already been trained")
                return
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._training = True
                print("Start Training")

                t = Thread(target=self.show, name='show')
                t.start()

                args = (self._env, self._policy, self._value, self._conv_net)
                kwargs = {'epochs':50, 'frames_per_epoch':2000, 'device':device, 'gif_epochs':1,
                          'model_name':self._model_name}
                ppo(*args, **kwargs)

                self._training = False
                self._done_training = True
                t.join()
            except:
                raise
            finally:
                self._training = False
                self._done_training = True
                self._save_weights()

    @staticmethod
    def stop_button():
        # make stop button
        @widgets.interact_manual
        def stop():
            raise KeyboardInterrupt

    def show(self):
        base_markdown = '![]({})'
        base_url = 'models/{}/gifs/{}.gif'.format(self._model_name, '{}')
        i = 0
        while self._training == True:
            clear_output()
            url, i = _next_url(base_url, i)
            if i >= 0:
                display(Markdown('### Epoch {}'.format(i)))
                display(Markdown(base_markdown.format(url)))
            else:
                i=0
            time.sleep(10)

    def _save_weights(self):
        weights = {}
        weights['policy'] = self._policy.state_dict()
        weights['value'] = self._value.state_dict()
        weights['conv'] = self._conv_net.state_dict()
        filepath = os.path.join(models_dir, self._model_name, 'weights.torch')
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)

    def _load_weights(self, name):
        filepath = os.path.join(models_dir, name, 'weights.torch')
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        self._policy.load_state_dict(weights['policy'])
        self._value.load_state_dict(weights['value'])
        self._conv_net.load_state_dict(weights['conv'])


def _next_url(base_url, i):
    url = base_url.format(i)
    while os.path.exists(url):
        i += 1
        url = base_url.format(i)
    i -= 1
    url = base_url.format(i)
    return url, i

def test_widgets():
    @widgets.interact_manual(freq=(0.,4.),
        color=['blue', 'red', 'green'], lw=(1., 10.))
    def plot(freq=1., color='blue', lw=2, grid=True):
        t = np.linspace(-1., +1., 1000)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(t, np.sin(2 * np.pi * freq * t),
                lw=lw, color=color)
        ax.grid(grid)
