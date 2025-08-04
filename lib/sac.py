# import argparse
import itertools
# from collections import namedtuple
# from itertools import count

import os
import numpy as np

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Independent, Normal
from typing import Union
from tensorboardX import SummaryWriter
from lib.utils import to_tensor

'''
Implementation of soft actor critic, dual Q network version 
Original paper: https://arxiv.org/abs/1801.01290
Not the author's implementation !
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Actor(nn.Module):
    def __init__(self, max_action, action_dim, state_dim, hidden1, hidden2, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.mu_head = nn.Linear(hidden2, action_dim)
        self.log_std_head = nn.Linear(hidden2, action_dim)
        self.max_action = max_action

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, state_dim, hidden1, hidden2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1, hidden2):
        super(Q, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Replay_buffer():
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_pool = torch.zeros(self.capacity, state_dim).float().to(device)
        self.action_pool = torch.zeros(self.capacity, action_dim).float().to(device)
        self.reward_pool = torch.zeros(self.capacity, 1).float().to(device)
        self.next_state_pool = torch.zeros(self.capacity, state_dim).float().to(device)
        self.done_pool = torch.zeros(self.capacity, 1).float().to(device)
        self.num_transition = 0

    def push(self, s, a, r, s_, d):
        index = self.num_transition % self.capacity
        s = torch.tensor(s).float().to(device)
        a = torch.tensor(a).float().to(device)
        r = torch.tensor(r).float().to(device)
        s_ = torch.tensor(s_).float().to(device)
        d = torch.tensor(d).float().to(device)
        for pool, ele in zip([self.state_pool, self.action_pool, self.reward_pool, self.next_state_pool, self.done_pool],
                           [s, a, r, s_, d]):
            pool[index] = ele
        self.num_transition += 1

    def sample(self, batch_size):
        index = np.random.choice(range(min(self.capacity,self.num_transition)), batch_size)
        bn_s, bn_a, bn_r, bn_s_, bn_d = self.state_pool[index], self.action_pool[index], self.reward_pool[index],\
                                        self.next_state_pool[index], self.done_pool[index]

        return bn_s, bn_a, bn_r, bn_s_, bn_d

class SAC():
    def __init__(self, action_dim, state_dim, hidden1, hidden2, capacity, lr, gradient_steps, batch_size, gamma, tau,
                 alpha, init_delta, delta_decay, warmup):
        super(SAC, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        # self.policy_net = Actor(1, action_dim, state_dim, hidden1, hidden2).to(device)
        self.policy_mu = nn.Parameter(torch.ones([action_dim], device=device))
        self.policy_log_sigma = nn.Parameter(torch.zeros([action_dim], device=device))

        self.value_net = Critic(state_dim, hidden1, hidden2).to(device)
        self.Target_value_net = Critic(state_dim, hidden1, hidden2).to(device)
        self.Q_net1 = Q(state_dim, action_dim, hidden1, hidden2).to(device)
        self.Q_net2 = Q(state_dim, action_dim, hidden1, hidden2).to(device)

        self.optimizer = optim.Adam(
            itertools.chain(
                [self.policy_mu],
                [self.policy_log_sigma],
                self.value_net.parameters(),
                self.Q_net1.parameters(),
                self.Q_net2.parameters(),
            ), lr=lr
        )

        self.capacity = capacity
        # self._alpha = 0.2
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = Replay_buffer(self.capacity, self.state_dim, self.action_dim)
        self.num_transition = 0  # pointer of replay buffer
        self.num_training = 1

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        # noise
        self.init_delta = init_delta
        self.delta_decay = delta_decay
        self.warmup = warmup


        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()
        self.min_Val = torch.tensor(1e-7).float().to(device)
        self.__eps = np.finfo(np.float32).eps.item()

        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        os.makedirs('./SAC_model/', exist_ok=True)

    def random_action(self, lbound, rbound):
        action = np.random.uniform(lbound, rbound, self.action_dim)
        return np.array(action)

    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        from scipy import stats
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)

    def select_action(self, state, lbound, rbound, episode):
        mu, log_sigma = self.policy_mu, self.policy_log_sigma
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action 


    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_mu.repeat(state.shape[0], 1), self.policy_log_sigma.repeat(state.shape[0], 1)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)
        z = noise.sample()
        action = torch.tanh(batch_mu + batch_sigma * z.to(device))
        log_prob = dist.log_prob(batch_mu + batch_sigma * z.to(device)) - torch.log(1 - action.pow(2) + self.min_Val)
        log_prob = torch.sum(log_prob, dim=-1).unsqueeze(-1)
        return action, log_prob

    def update(self):
        for _ in range(self.gradient_steps):
            bn_s, bn_a, bn_r, bn_s_, bn_d = self.replay_buffer.sample(self.batch_size)

            # # normalize the reward
            # batch_mean_reward = torch.mean(bn_r)
            # if self.moving_average is None:
            #     self.moving_average = batch_mean_reward
            # else:
            #     self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
            # bn_r -= self.moving_average

            target_value = self.Target_value_net(bn_s_)

            next_q_value = bn_r + (1 - bn_d) * self.gamma * target_value

            excepted_value = self.value_net(bn_s)
            excepted_Q1 = self.Q_net1(bn_s, bn_a)
            excepted_Q2 = self.Q_net2(bn_s, bn_a)
            sample_action, log_prob = self.evaluate(bn_s)

            excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
            next_value = excepted_new_Q - self._alpha*log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)

            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V

            # Dual Q net
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean()  # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()

            # actor loss
            pi_loss = (self._alpha*log_prob - excepted_new_Q).mean()  # according to original paper


            # mini batch gradient descent
            self.optimizer.zero_grad()
            loss = V_loss + Q1_loss + pi_loss + Q2_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.policy_mu, 0.5)
            nn.utils.clip_grad_norm_(self.policy_log_sigma, 0.5)
            self.optimizer.step()

            if self._is_auto_alpha:
                log_prob = log_prob.detach() + self._target_entropy
                # please take a look at issue #258 if you'd like to change this line
                alpha_loss = -(self._log_alpha * log_prob).mean()
                self._alpha_optim.zero_grad()
                alpha_loss.backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()

            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

            self.num_training += 1

            result = {
                'Loss/V_loss': V_loss,
                'Loss/Q1_loss': Q1_loss,
                'Loss/Q2_loss': Q2_loss,
                'Loss/policy_loss': pi_loss,
            }

            if self._is_auto_alpha:
                result["loss/alpha"] = alpha_loss.item()
                result["alpha"] = self._alpha.item()  # type: ignore

            return result

    def save(self, output):
        # torch.save(self.policy_net.state_dict(), '{}/policy_net.pth'.format(output))
        torch.save(self.value_net.state_dict(), '{}/value_net.pth'.format(output))
        torch.save(self.Q_net1.state_dict(), '{}/Q_net1.pth'.format(output))
        torch.save(self.Q_net2.state_dict(), '{}/Q_net2.pth'.format(output))
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self, output):
        # self.policy_net.load_state_dict(torch.load('{}/policy_net.pth').format(output))
        self.value_net.load_state_dict(torch.load('{}/value_net.pth').format(output))
        self.Q_net1.load_state_dict(torch.load('{}/Q_net1.pth').format(output))
        self.Q_net2.load_state_dict(torch.load('{}/Q_net2.pth').format(output))
        print("model has been load")
