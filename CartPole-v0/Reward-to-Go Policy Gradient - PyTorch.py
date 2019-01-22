# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym.spaces import Discrete, Box

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class Policy(nn.Module):
    def __init__(self,sizes, activation=torch.tanh, output_activation=None):
        super(Policy,self).__init__()
        self.sizes = sizes
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        for i in range(len(sizes)-1):
            self.layers.append(nn.Linear(in_features = sizes[i], out_features = sizes[i+1]))
        self.myparameters = [layer.parameters() for layer in self.layers]
    
    def forward(self,x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        x = self.output_activation(x) if self.output_activation!=None else x
        return x
  

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    
    policy = Policy([obs_dim]+hidden_sizes+[n_acts])
    optimizer = optim.Adam(policy.parameters(), lr = lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            pred = policy(torch.tensor(obs,dtype=torch.float32).detach())
            dist = F.softmax(pred,dim=0).data
            act = int(torch.multinomial(dist,1)[0])
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        actions = policy(torch.tensor(batch_obs,dtype=torch.float32))
        action_masks = torch.eye(n_acts)[batch_acts].float()
        log_probs = torch.tensor(action_masks,dtype=torch.float32)*nn.LogSoftmax(dim=1)(actions)
        loss = -torch.mean(torch.tensor(batch_weights)*torch.sum(log_probs,dim=1))
        loss.backward()     
        optimizer.step()
        
        return loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)