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

class ModelManageBase:
    @staticmethod
    def _is_used(name):
        required_files = ['gifs','latest_run.csv','weights.torch']
        required_files = set(required_files)
        path = os.path.join(models_dir, name)
        if os.path.isdir(path):
            if required_files.issubset(set(os.listdir(path))):
                return True
        return False

    def _get_used_names(self):
        ret = ['']
        for name in os.listdir(models_dir):
            if self._is_used(name):
                ret.append(name)
        return ret

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

    def _initialize_nets(self):
        latent_size = 100
        hidden_size = 100
        layers = 4
        action_size = self._env.action_space.n

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._conv_net = ConvNetwork(latent_size)
        self._value = MLP(latent_size, hidden_size, layers, action_size)
        self._policy = Actor(MLP(latent_size, hidden_size, layers, action_size))

        self._conv_net.to(self.device)
        self._value.to(self.device)
        self._policy.to(self.device)

class ModelBuilder(ModelManageBase):
    def __init__(self):
        self._env = get_env()

        self._name_assigned = False
        self._reward_assignd = False
        self._model_built = False
        self._training = False
        self._done_training = False

    def set_reward(self, reward):
        self._reward_assigned = True
        self._reward = reward
        self._env.update_reward_func(reward)

    def run_builder_tool(self):
        display(Markdown("## Choose Model Name"))
        @widgets.interact_manual
        def name_model(name=''):
            cur_model_dir = os.path.join(models_dir, name)
            dir_exists = os.path.isdir(cur_model_dir)
            if ('' != name) and (not self._is_used(name)):
                self._name_assigned = True
                if not dir_exists:
                    os.mkdir(cur_model_dir)
                self._model_name = name
                print("{} is a valid unique model name.".format(name))
            else:
                print("Enter a valid new name. Existing model names: {}".format(
                    '\n'.join(self._get_used_names())))

        display(Markdown("## Build Initial Model\n ### Leave empty for random initialization."))
        @widgets.interact_manual(clone_from=self._get_used_names())
        def build(clone_from=''):
            if not self._reward_assigned:
                print("Must create reward function before running.")
                return
            if not self._name_assigned:
                print("Must choose model name before running.")
                return

            self._initialize_nets()
            self._model_built = True
            if clone_from == '':
                print("New model initialized with random parameters.")
            else:
                self._load_weights(clone_from)
                print("New model cloned from {}.".format(clone_from))

        display(Markdown("## Choose Hyperparameters/Train Model"))
        @widgets.interact_manual(frames_per_epoch=(400,4000),
                                 epochs=(1,100),
                                 max_episode_len=(50,500),
                                 batch_size=[32,64,128,256])
        def run(frames_per_epoch=1000, epochs=30, max_episode_len=200,batch_size=64):
            if not self._model_built:
                print("Must build the model first.")
                return
            if self._done_training:
                print("This model has already been trained.")
                return
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._training = True
                print("Start Training")

                t = Thread(target=self.show, args=(epochs,), name='show')
                t.start()

                args = (self._env, self._policy, self._value, self._conv_net)
                kwargs = {'epochs':epochs, 'frames_per_epoch':frames_per_epoch,
                          'device':device, 'gif_epochs':1,'batch_size':batch_size,
                          'model_name':self._model_name,'max_episode_length':max_episode_len}
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

    def show(self, epochs):
        base_markdown = '![]({})'
        base_url = 'models/{}/gifs/{}.gif'.format(self._model_name, '{}')
        i = 0
        final_update = False
        first_update = True
        while self._training == True and (not final_update):
            final_update = not self._training
#             url, i2 = _next_url(base_url, i)
            rewards = self._get_rewards()
            i2 = len(rewards)
            url = _next_url(base_url, i2)
            if ((i2 > i) or first_update) and (url is not None):
                first_update = False
                i = i2
                time.sleep(1) # wait to make sure file is completed
                clear_output()
                if i >= 0:
                    display(Markdown('### Epoch {}'.format(i)))
                    display(Markdown(base_markdown.format(url)))

                    if rewards:
                        plt.plot(rewards, 'b-o')
                        plt.xlim(0,epochs)
                        plt.ylim(0,1.1*max(rewards))
                        plt.ylabel('Average Reward')
                        plt.xlabel('Epoch')
                        plt.show()
                else:
                    i=0
            time.sleep(1)

    def _get_rewards(self):
        filepath = os.path.join(models_dir, self._model_name, 'latest_run.csv')
        rewards = []
        try:
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    r,_,_ = line.partition(',')
                    try:
                        rewards.append(float(r))
                    except:
                        pass
        except:
            pass
        return rewards

class ModelTester(ModelManageBase):
    def __init__(self):
        self._get_env()

    def run_test_tool(self):
        self._get_model()
        self._get_run()

    def _get_env(self):
        self._env = get_env()
        self._env.update_random_settings({'car_rand_loc':False})

    def _get_model(self):
        @widgets.interact_manual(name=self._get_used_names())
        def _test_model(name=''):
            if name == '':
                print("Choose a trained model")
                return
            else:
                self._name = name
                self._initialize_nets()
                self._load_weights(name)
                print("{} is loaded".format(name))

    def _get_run(self):
        @widgets.interact_manual
        def _run_env(real_time=False):
            rollout = []
            max_episode_len = 300
            state = self._env.reset()
            image = _prepare_numpy(state[0], self.device)
            tot_reward = 0
            for p in self._conv_net.parameters():
                p.require_grad = False
            for p in self._policy.parameters():
                p.require_grad = False
            for _ in range(max_episode_len):
                latent_state = self._conv_net(image)
                action_dist, action = self._policy(latent_state, True)
                state, r, done, _ = self._env.step(int(action[0,0].item()))
                tot_reward += r
                image = state[0]
                rollout.append((image,r))
                image = _prepare_numpy(image, self.device)
                if done: break
                if real_time:
                    time.sleep(1/self._env._fps)

            for p in self._conv_net.parameters():
                p.require_grad = True
            for p in self._policy.parameters():
                p.require_grad = True

            tot_time = len(rollout) / self._env._fps
            print("Total Reward: {:2.2f}".format(tot_reward))
            print("Total time: {:2.2f} seconds".format(tot_time))
#             return rollout, tot_reward

def _next_url(base_url, i):
    url = base_url.format(i)
    if os.path.exists(url):
        return url
    else:
        return None

# def _next_url(base_url, i):
#     url = base_url.format(i)
#     while os.path.exists(url):
#         i += 1
#         url = base_url.format(i)
#     i -= 1
#     url = base_url.format(i)
#     if os.path.exists(url):
#         return url, i
#     else:
#         return None, 0

def _prepare_numpy(ndarray, device):
    return torch.from_numpy(ndarray).float().unsqueeze(0).to(device)
