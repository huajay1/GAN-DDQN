import torch, time, os, pickle, glob, math, json
import numpy as np
import csv
from timeit import default_timer as timer 
from datetime import timedelta
import itertools
import pandas as pd

# simulation environment
from cellular_env import cellularEnv

import matplotlib
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import torchvision.transforms as T

from utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory
# from utils.wrappers import *
from utils.utils import initialize_weights


#%%
class Generator(nn.Module):
    def __init__(self, state_size, num_actions, num_samples, embedding_dim):
        super(Generator, self).__init__()
        
        self.state_size = state_size # input_shape --> num_channels * hight * width 
        self.num_actions = num_actions
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda')

        self.embed_layer_1 = nn.Linear(self.state_size, self.embedding_dim)
        # self.embed_layer_drop_1 = nn.Dropout(0.5)
        self.embed_layer_2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.embed_layer_drop_2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.embedding_dim, 256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, self.num_actions)

        initialize_weights(self)

    def forward(self, x, tau):   # tau: torch.randn(self.batch_size, self.num_samples)
        state_tile = x.repeat(1, self.num_samples)    # [self.batch_size, (self.state_size * self.num_samples)]
        state_reshape = state_tile.view(-1, self.state_size)   
        state = F.relu(self.embed_layer_1(state_reshape))       # [(self.batch_size * self.num_samples), self.embedding_dim]
        # state = self.embed_layer_drop_1(state)

        tau = tau.view(-1, 1)
        pi_mtx = torch.from_numpy(np.expand_dims(np.pi * np.arange(0, self.embedding_dim), axis=0)).to(torch.float).to(self.device)
        cos_tau = torch.cos(torch.matmul(tau, pi_mtx)) # [(self.batch_size * self.num_samples), self.embedding_dim]
        pi = F.relu(self.embed_layer_2(cos_tau))  # [(self.batch_size * self.num_samples), self.embedding_dim]
        # pi = self.embed_layer_drop_2(pi)

        x = state * pi
        x = F.relu(self.fc1(x))
        # x = self.drop1(x)
        x = F.relu(self.fc2(x))
        # x = self.drop2(x)
        x = self.fc3(x)    # [(self.batch_size * self.num_samples), self.num_actions]

        net = torch.transpose(x.view(-1, self.num_samples, self.num_actions), 1, 2)  # [self.batch_size, self.num_actions, self.num_samples]
        
        return net
    
#%%
class Discriminator(nn.Module):
    def __init__(self, num_samples, num_outputs):
        super(Discriminator, self).__init__()

        self.num_inputs = num_samples
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(self.num_inputs, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.num_outputs)

        initialize_weights(self)

    def forward(self, x, z):
        # add little noise
        x = x + z
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, start_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.start_timesteps = start_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        if t < self.start_timesteps:
            return self.initial_p
        else:
            fraction = min(float(t) / (self.schedule_timesteps + self.start_timesteps), 1.0)
            return self.initial_p + fraction * (self.final_p - self.initial_p)

class WGAN_GP_Agent(object):
    def __init__(self, static_policy, num_input, num_actions):
        super(WGAN_GP_Agent, self).__init__()
        # parameters
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda')

        self.gamma = 0.75
        self.lr_G = 1e-4
        self.lr_D = 1e-4
        self.target_net_update_freq = 10
        self.experience_replay_size = 2000
        self.batch_size = 32
        self.update_freq = 200
        self.learn_start = 0
        self.tau = 0.1                        # default is 0.005

        self.static_policy = False
        self.num_feats = num_input
        self.num_actions = num_actions
        self.z_dim = 32
        self.num_samples = 32

        self.lambda_ = 10
        self.n_critic = 5  # the number of iterations of the critic per generator iteration1
        self.n_gen = 1

        self.declare_networks()

        self.G_target_model.load_state_dict(self.G_model.state_dict())
        self.G_optimizer = optim.Adam(self.G_model.parameters(), lr=self.lr_G, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D_model.parameters(), lr=self.lr_D, betas=(0.5, 0.999))

        self.G_model = self.G_model.to(self.device)
        self.G_target_model = self.G_target_model.to(self.device)
        self.D_model = self.D_model.to(self.device)

        if self.static_policy:
            self.G_model.eval()
            self.D_model.eval()
        else:
            self.G_model.train()
            self.D_model.train()

        self.update_count = 0
        self.nsteps = 1
        self.nstep_buffer = []

        self.declare_memory()

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        
        self.one = torch.tensor([1], device=self.device, dtype=torch.float)
        self.mone = self.one * -1

        self.batch_normalization = nn.BatchNorm1d(self.batch_size).to(self.device)

    def declare_networks(self):
        # Output the probability of each sample
        self.G_model = Generator(self.num_feats, self.num_actions, self.num_samples, self.z_dim) # output: batch_size x (num_actions*num_samples)
        self.G_target_model = Generator(self.num_feats, self.num_actions, self.num_samples, self.z_dim)
        self.D_model = Discriminator(self.num_samples, 1) # input: batch_size x num_samples output: batch_size

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def save_w(self):
            if not os.path.exists('./saved_agents/GANDDQN'):
                os.makedirs('./saved_agents/GANDDQN')
            torch.save(self.G_model.state_dict(), './saved_agents/GANDDQN/G_model_10M_0.01.dump')
            torch.save(self.D_model.state_dict(), './saved_agents/GANDDQN/D_model_10M_0.01.dump')

    def save_replay(self):
        pickle.dump(self.memory, open('./saved_agents/exp_replay_agent.dump', 'wb'))

    def load_replay(self):
        fname = './saved_agents/exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    def load_w(self):
        fname_G_model = './saved_agents/G_model_0.dump'
        fname_D_model = './saved_agents/D_model_0.dump'

        if os.path.isfile(fname_G_model):
            self.G_model.load_state_dict(torch.load(fname_G_model))
            self.G_target_model.load_state_dict(self.G_model.state_dict())
        
        if os.path.isfile(fname_D_model):
            self.D_model.load_state_dict(torch.load(fname_D_model))

    def plot_loss(self):
        plt.figure(2)
        plt.clf()
        plt.title('Training loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(self.train_hist['G_loss'], 'r')
        plt.plot(self.train_hist['D_loss'], 'b')
        plt.legend(['G_loss', 'D_loss'])
        plt.pause(0.001)

    def prep_minibatch(self, prev_t, t):
        transitions = self.memory.determine_sample(prev_t, t)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        batch_state = torch.tensor(batch_state).to(torch.float).to(self.device)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(-1, 1)
        batch_next_state = torch.tensor(batch_next_state).to(torch.float).to(self.device)

        return batch_state, batch_action, batch_reward, batch_next_state

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            for target_param, param in zip(self.G_target_model.parameters(), self.G_model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def get_max_next_state_action(self, next_states, noise):
        samples = self.G_target_model(next_states, noise)
        return samples.mean(2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.num_samples)

    def calc_gradient_penalty(self, real_data, fake_data, noise):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size()).to(self.device)
        interpolates = alpha * real_data.data + (1 - alpha) * fake_data.data
        interpolates.requires_grad = True

        disc_interpolates = self.D_model(interpolates, noise)
        gradients = grad(outputs=disc_interpolates, inputs=interpolates, 
                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_
        return gradient_penalty

    def adjust_G_lr(self, epoch):
        lr = self.lr_G * (0.1 ** (epoch // 3000))
        for param_group in self.G_optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_D_lr(self, epoch):
        lr = self.lr_D * (0.1 ** (epoch // 3000))
        for param_group in self.D_optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, frame=0):
        if self.static_policy:
            return None
        
        # self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start:
            return None

        if frame % self.update_freq != 0:
            return None

        if self.memory.__len__() != self.experience_replay_size:
            return None
        
        print('Training.........')

        self.adjust_G_lr(frame)
        self.adjust_D_lr(frame)

        self.memory.shuffle_memory()
        len_memory = self.memory.__len__()
        memory_idx = range(len_memory)
        slicing_idx = [i for i in memory_idx[::self.batch_size]]
        slicing_idx.append(len_memory)
        # print(slicing_idx)

        self.G_model.eval()
        for t in range(len_memory // self.batch_size):
            for _ in range(self.n_critic):
                # update Discriminator
                batch_vars = self.prep_minibatch(slicing_idx[t], slicing_idx[t+1])
                batch_state, batch_action, batch_reward, batch_next_state = batch_vars
                G_noise = (torch.rand(self.batch_size, self.num_samples)).to(self.device)

                batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.num_samples)

                # estimate
                current_q_values_samples = self.G_model(batch_state, G_noise) # batch_size x (num_actions*num_samples)
                current_q_values_samples = current_q_values_samples.gather(1, batch_action).squeeze(1)

                # target
                with torch.no_grad():
                    expected_q_values_samples = torch.zeros((self.batch_size, self.num_samples), device=self.device, dtype=torch.float) 
                    max_next_action = self.get_max_next_state_action(batch_next_state, G_noise)
                    expected_q_values_samples = self.G_model(batch_next_state, G_noise).gather(1, max_next_action).squeeze(1)
                    expected_q_values_samples = batch_reward + self.gamma * expected_q_values_samples

                D_noise = 0. * torch.randn(self.batch_size, self.num_samples).to(self.device)
                # WGAN-GP
                self.D_model.zero_grad()
                D_real = self.D_model(expected_q_values_samples, D_noise)
                D_real_loss = torch.mean(D_real)

                D_fake = self.D_model(current_q_values_samples, D_noise)
                D_fake_loss = torch.mean(D_fake)

                gradient_penalty = self.calc_gradient_penalty(expected_q_values_samples, current_q_values_samples, D_noise)
                
                D_loss = D_fake_loss - D_real_loss + gradient_penalty

                D_loss.backward()
                self.D_optimizer.step()

            # update G network
            self.G_model.train()
            self.G_model.zero_grad()

            # estimate
            current_q_values_samples = self.G_model(batch_state, G_noise) # batch_size x (num_actions*num_samples)
            current_q_values_samples = current_q_values_samples.gather(1, batch_action).squeeze(1)
            
            # WGAN-GP
            D_fake = self.D_model(current_q_values_samples, D_noise)
            G_loss = -torch.mean(D_fake)
            G_loss.backward()
            for param in self.G_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.G_optimizer.step()

            self.train_hist['G_loss'].append(G_loss.item())
            self.train_hist['D_loss'].append(D_loss.item())

            self.update_target_model()

        print('current q value', current_q_values_samples.mean(1))
        print('expected q value', expected_q_values_samples.mean(1))

    
def action_space(total, num):
    tmp = list(itertools.product(range(total + 1), repeat=num))
    result = []
    for value in tmp:
        if sum(value) == total:
            result.append(list(value))
    result = np.array(result)
    [i, j] = np.where(result == 0)
    result = np.delete(result, i, axis=0)
    print(result.shape)

    return result

def state_update(state, ser_cat):
    discrete_state = np.zeros(state.shape)
    if state.all() == 0:
        return discrete_state
    for ser_name in ser_cat:
        ser_index = ser_cat.index(ser_name)
        discrete_state[ser_index] = state[ser_index]
    discrete_state = (discrete_state-discrete_state.mean())/discrete_state.std()
    return discrete_state

# def calc_reward(qoe, se, low, high):
#     utility = np.matmul(qoe_weight, qoe.reshape((3, 1))) + se_weight * se
#     if utility < low:
#         reward = -1
#     elif utility > high:
#         reward = 1
#     else:
#         reward = 0
#     return utility, reward
def calc_reward(qoe, se, threshold):
    utility = np.matmul(qoe_weight, qoe.reshape((3, 1))) + se_weight * se
    if threshold > 5.5:
        threshold = 5.5
    if utility < threshold:
        reward = 0
    else:
        reward = 1
    return utility, reward

def get_action(model, s, z, eps, device):
    if np.random.random() >= eps:
        X = torch.tensor(s).unsqueeze(0).to(torch.float).to(device)
        a = model.G_model(X, z).squeeze(0).mean(1).max(0)[1]
        # print(a)  
        return a.item()
    else:
        return np.random.randint(0, model.num_actions)

def plot_rewards(rewards):
    plt.figure(1)
    plt.clf()
    rewards = np.array(rewards)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards.cumsum())
    plt.pause(0.001)


# torch.cuda.manual_seed(100)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda')
# exploration_fraction = 0.3
# exploration_start = 0.
total_timesteps = 10000
# exploration_final_eps = 0.02
#epsilon variables

epsilon_start    = 1.0
epsilon_final    = 0.01
epsilon_decay    = 3000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# parameters of celluar environment
ser_cat_vec = ['volte', 'embb_general', 'urllc']
band_whole_no = 10 * 10**6
band_per = 1 * 10**6
qoe_weight = [1, 1, 1]
se_weight = 0.01
dl_mimo = 64
learning_window = 2000

# Create the schedule for exploration starting from 1.
# exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
#                                 start_timesteps = int(total_timesteps * exploration_start),
#                                 initial_p=1.0,
#                                 final_p=exploration_final_eps)

plt.ion()

env = cellularEnv(ser_cat=ser_cat_vec, learning_windows=learning_window, dl_mimo=dl_mimo)
action_space = action_space(10, 3) * band_per
num_actions = len(action_space)
print(num_actions)

model = WGAN_GP_Agent(static_policy=False, num_input=3, num_actions=num_actions)
# model.load_w()
G_noise = (torch.rand(1, model.num_samples)).to(device)
env.countReset()
env.activity()
observation = state_update(env.tx_pkt_no, env.ser_cat)
print(observation)
# observation = torch.from_numpy(observation).unsqueeze(0).to(torch.float)

log = {}
rewards = []
uyilitys = [0.]
observations = []
actions = []
SE = []
QoE = []

for frame in range(1, total_timesteps + 1):
    # env.render()
    # epsilon = exploration.value(t)
    epsilon = epsilon_by_frame(frame)
    # Select and perform an action
    observations.append(observation.tolist())
    action = get_action(model, observation, G_noise, epsilon, device)
    actions.append(action)
    env.band_ser_cat = action_space[action]
    prev_observation = observation

    for i in itertools.count():
        env.scheduling()
        env.provisioning()
        if i == (learning_window - 1):
            break
        else:
            env.bufferClear()
            env.activity()
    
    qoe, se = env.get_reward()
    # utility, reward = calc_reward(qoe, se, 3, 5.7)
    threshold = 3.5 + 1.5 * frame / (total_timesteps/1.5)
    print(threshold)
    utility, reward = calc_reward(qoe, se, threshold)
    QoE.append(qoe.tolist())
    SE.append(se[0])
    rewards.append(reward)
    uyilitys.append(utility)

    observation = state_update(env.tx_pkt_no, env.ser_cat)
    print(observation)
    # observation = torch.from_numpy(observation).unsqueeze(0).to(torch.float)

    model.append_to_replay(prev_observation, action, reward, observation)
    model.update(frame)
    env.countReset()
    env.activity()
    print('GANDDQN=====episode: %d, epsilon: %.3f, utility: %.5f, reward: %.5f' % (frame, epsilon, utility, reward))
    print('qoe', qoe)
    print('bandwidth-allocation solution', action_space[action])

    # plot_rewards(rewards)

    if frame % 200 == 0:
        print('frame index [%d], epsilon [%.4f]' % (frame, epsilon))
        model.save_w()
        log['state'] = observations
        log['action'] = actions
        log['SE'] = SE
        log['QoE'] = QoE
        log['reward'] = rewards

        f = open('./log/GANDDQN/log_10M_1M_LURLLC.txt', 'w')
        f.write(json.dumps(log))
        f.close()
    
print('Complete')
plt.ioff()
plt.show()