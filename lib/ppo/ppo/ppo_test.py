import numpy as np
import torch
from torch import nn

from ppo import MLP, PPO, Actor, Critic, Gaussian



mlp = MLP([128, 128, 128], nn.ReLU, 10, 1)
explorer = Gaussian(1, 1.0)
actor = Actor(mlp, explorer)
mlp2 = MLP([128, 128, 128], nn.ReLU, 10, 1)
critic = Critic(mlp2)

ppo = PPO(actor, critic, 1, 24, 4, 1)
# obs = env.reset()
obs = np.zeros([1, 10], dtype=np.float32)
for i in range(24):
    action = ppo.act(obs)
    print(action)
    # next_obs, rews, dones, timeout = env.step(obs)
    next_obs = obs
    rews = np.array([0.])
    dones = np.array([0], dtype=bool)
    timeout = np.array([0], dtype=bool)
    ppo.step(obs, rews, dones, timeout)

ppo.update()