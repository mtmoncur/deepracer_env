import os
from itertools import chain
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np

class ExperienceDataset(Dataset):
    def __init__(self, experience):
        super(ExperienceDataset, self).__init__()
        self._exp = []
        for x in experience:
            self._exp.extend(x)
        self._length = len(self._exp)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return self._length

def multinomial_likelihood(dist, idx):
    return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)

def ppo(environment, policy, value, embedding_net, epochs=100,
        frames_per_epoch=2000, max_episode_length=200, gamma=0.99, policy_epochs=3, batch_size=64, epsilon=0.2,
        data_loader_threads=1, device=torch.device('cpu'), lr=1e-3, betas=(0.9, 0.999),
        weight_decay=0.01, gif_name='', gif_epochs=0, model_name='project', csv_file='latest_run.csv'):
    # Set up experiment details
    models_path = os.path.join('models', model_name)
    if not os.path.isdir(models_path):
        os.makedirs(models_path)
    if gif_epochs:
        gif_path = os.path.join(models_path, 'gifs')
        if not os.path.isdir(gif_path):
            os.mkdir(gif_path)
    csv_file = os.path.join(models_path, csv_file)

    # Clear the csv file
    with open(csv_file, 'w') as f:
        f.write('avg_reward, value_loss, policy_loss\n')

    # Move networks to the correct device
    policy = policy.to(device)
    value = value.to(device)
    embedding_net = embedding_net.to(device)

    # Collect parameters
    params = chain(policy.parameters(), value.parameters(),
                  embedding_net.parameters())

    # Set up optimization
    optimizer = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    value_criteria = nn.MSELoss()

    # Calculate the upper and lower bound for PPO
    ppo_lower_bound = 1 - epsilon
    ppo_upper_bound = 1 + epsilon

#     loop = tqdm(total=epochs, position=0, leave=False)

    for e in range(epochs):
        # Run the environments
        rollouts, rewards = _run_envs(environment,
                                      embedding_net,
                                      policy,
                                      frames_per_epoch,
                                      max_episode_length,
                                      gamma,
                                      device)

        # Collect the experience
        avg_r = sum(rewards) / len(rewards)
#         loop.set_description('avg reward: % 6.2f' % avg_r)

        # Make gifs
        if gif_epochs and e % gif_epochs == 0:
            _make_gif(rollouts[-1], os.path.join(gif_path, gif_name + '%d.gif' % e))

        # Update the policy
        experience_dataset = ExperienceDataset(rollouts)
        data_loader = DataLoader(experience_dataset, num_workers=data_loader_threads, batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)
        avg_policy_loss = 0
        avg_val_loss = 0
        for _ in range(policy_epochs):
            avg_policy_loss = 0
            avg_val_loss = 0
            for images, old_action_dist, old_action, reward, ret in data_loader:
                images = _prepare_tensor_batch(images, device)
                old_action_dist = _prepare_tensor_batch(old_action_dist, device)
                old_action = _prepare_tensor_batch(old_action, device)
                ret = _prepare_tensor_batch(ret, device).unsqueeze(1)

                optimizer.zero_grad()

                latent_state = embedding_net(images)

                # Calculate the ratio term
                current_action_dist = policy(latent_state, False)
                current_likelihood = multinomial_likelihood(current_action_dist, old_action)
                old_likelihood = multinomial_likelihood(old_action_dist, old_action)
                ratio = (current_likelihood / old_likelihood)

                # Calculate the value loss
                expected_returns = value(latent_state)
                val_loss = value_criteria(expected_returns, ret)

                # Calculate the policy loss
                advantage = ret - expected_returns.detach()
                lhs = ratio * advantage
                rhs = torch.clamp(ratio, ppo_lower_bound, ppo_upper_bound) * advantage
                policy_loss = -torch.mean(torch.min(lhs, rhs))

                # For logging
                avg_val_loss += val_loss.item()
                avg_policy_loss += policy_loss.item()

                # Backpropagate
                loss = policy_loss + val_loss
                loss.backward()
                optimizer.step()

            # Log info
            avg_val_loss /= len(data_loader)
            avg_policy_loss /= len(data_loader)
#             loop.set_description(
#                 'avg reward: % 6.2f, value loss: % 6.2f, policy loss: % 6.2f' % (avg_r, avg_val_loss, avg_policy_loss))
        with open(csv_file, 'a+') as f:
            f.write('%6.2f, %6.2f, %6.2f\n' % (avg_r, avg_val_loss, avg_policy_loss))
#         print()
#         loop.update(1)

def _calculate_returns(trajectory, gamma):
    current_return = 0
    for i in reversed(range(len(trajectory))):
        state, action_dist, action, reward = trajectory[i]
        ret = reward + gamma * current_return
        trajectory[i] = (state, action_dist, action, reward, ret)
        current_return = ret


def _run_envs(env, embedding_net, policy, max_frames, max_episode_length,
              gamma, device):
    rollouts = []
    rewards = []
    frames = 0
    while frames < max_frames:
        current_rollout = []
        s = env.reset()
        image = s[0]
        episode_reward = 0
        for _ in range(max_episode_length):
            frames += 1
            input_state = _prepare_numpy(image, device)
            latent_state = embedding_net(input_state)
            
            action_dist, action = policy(latent_state)
            action_dist, action = action_dist[0], action[0]  # Remove the batch dimension
            s_prime, r, t, _ = env.step(int(action[0]))

            if type(r) != float:
                print('run envs:', r, type(r))

            current_rollout.append((image, action_dist.cpu().detach().numpy(), action, r))
            episode_reward += r
            if t:
                break
            s = s_prime
            image = s[0]
        _calculate_returns(current_rollout, gamma)
        rollouts.append(current_rollout)
        rewards.append(episode_reward)
    return rollouts, rewards

def _prepare_numpy(ndarray, device):
    return torch.from_numpy(ndarray).float().unsqueeze(0).to(device)

def _prepare_tensor_batch(tensor, device):
    return tensor.detach().float().to(device)

def _make_gif(rollout, filename):
    with imageio.get_writer(filename, mode='I', duration=1 / 15) as writer:
        for x in rollout:
            writer.append_data((x[0][:, :, :]*255).astype(np.uint8))