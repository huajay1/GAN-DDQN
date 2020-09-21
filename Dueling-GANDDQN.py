import torch, time, os, pickle, glob, math, json
import numpy as np
import csv
from timeit import default_timer as timer 
from datetime import timedelta
import itertools
import pandas as pd
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
    def __init__(self, device, num_actions, num_particles, state_size=3, noise_size=10):
        # device: 'cpu' or 'gpu';
        # state_size: the length of input state, default=3 
        # noise_size: the number of sampled noise, default=10
        # num_action: the number of actions
        # num_particles: the number of output particles
        super(Generator, self).__init__()
        self.state_size = state_size 
        self.noise_size = noise_size
        self.num_actions = num_actions
        self.num_particles = num_particles
        self.device = device

        self.embed_layer_state = nn.Linear(self.state_size, 256)
        self.embed_layer_noise = nn.Linear(self.noise_size, 256)

        # self.common_layer1 = nn.Linear(256, 256)
        # self.common_layer2 = nn.Linear(256, 256)
        self.common_layer = nn.Linear(256, 256)

        self.value_layer1 = nn.Linear(256, 128)
        self.value_layer2 = nn.Linear(128, num_particles)
        # self.value_layer = nn.Linear(256, num_particles)

        self.advantage_layer1 = nn.Linear(256, 128)
        self.advantage_layer2 = nn.Linear(128, num_actions)
        # self.advantage_layer = nn.Linear(256, num_actions)
        initialize_weights(self)

    def forward(self, state, noise):   
        # state: environment state, a 3-dimention vector 
        # noise: sampled noise, a noise_size-dimention vector
        state_embedding = F.relu(self.embed_layer_state(state))     # batch_size x 256
        noise_embedding = F.relu(self.embed_layer_noise(noise))     # batch_size x 256

        # element-wise product of embedded state and noise
        x = torch.mul(state_embedding, noise_embedding)
        # through common layers
        # x = F.relu(self.common_layer1(x))
        # x = F.relu(self.common_layer2(x))
        x = F.relu(self.common_layer(x))

        # output value vector, which is a set of particles coming from value distribution
        value = F.relu(self.value_layer1(x))  
        value = F.relu(self.value_layer2(value))    # batch_size x num_particles
        # value = F.relu(self.value_layer(x))

        # output advantage, which is a scale number for each action
        advantage = F.relu(self.advantage_layer1(x))  
        advantage = self.advantage_layer2(advantage)    #batch_size x num_actions
        # advantage = torch.tanh(self.advantage_layer(x))

        return value, advantage
    
#%%
class Discriminator(nn.Module):
    def __init__(self, num_particles):
        super(Discriminator, self).__init__()
        self.num_inputs = num_particles
        self.fc1 = nn.Linear(self.num_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        initialize_weights(self)

    def forward(self, value_particles):
        x = F.relu(self.fc1(value_particles))
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
    def __init__(self, device, state_size, noise_size, num_actions, num_particles):
        super(WGAN_GP_Agent, self).__init__()
        self.device = torch.device(device)
        self.gamma = 0.8
        self.lr_G = 1e-3
        self.lr_D = 1e-3
        self.lr_mse = 1e-3
        self.target_net_update_freq = 1
        self.experience_replay_size = 50000
        self.batch_size = 32
        self.update_freq = 1
        self.learn_start = 200
        self.tau = 0.01                        # default is 0.005
        self.state_size = state_size
        self.noise_size = noise_size
        self.num_actions = num_actions
        self.num_particles = num_particles

        self.lambda_ = 10
        self.n_critic = 5  # the number of iterations of the critic per generator iteration1
        self.n_gen = 1

        self.declare_networks()

        self.G_target_model.load_state_dict(self.G_model.state_dict())
        self.G_optimizer = optim.Adam(self.G_model.parameters(), lr=self.lr_G, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D_model.parameters(), lr=self.lr_D, betas=(0.5, 0.999))
        self.G_optimizer_MSE = optim.Adam(self.G_model.parameters(), lr=self.lr_mse)

        self.G_model = self.G_model.to(self.device)
        self.G_target_model = self.G_target_model.to(self.device)
        self.D_model = self.D_model.to(self.device)

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
        self.G_model = Generator(self.device, self.num_actions, self.num_particles, self.state_size, self.noise_size)
        self.G_target_model = Generator(self.device, self.num_actions, self.num_particles, self.state_size, self.noise_size)
        self.D_model = Discriminator(self.num_particles) 

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def save_w(self):
            if not os.path.exists('./saved_agents/Dueling_GANDDQN-v2'):
                os.makedirs('./saved_agents/Dueling_GANDDQN-v2')
            torch.save(self.G_model.state_dict(), './saved_agents/Dueling_GANDDQN/Dueling_GANDDQN_G-v2.dump')
            torch.save(self.D_model.state_dict(), './saved_agents/Dueling_GANDDQN/Dueling_GANDDQN_D-v2.dump')

    def save_replay(self):
        pickle.dump(self.memory, open('./saved_agents/exp_replay_agent.dump', 'wb'))

    def load_replay(self):
        fname = './saved_agents/exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    def load_w(self):
        fname_G_model = './saved_agents/Dueling_GANDDQN/Dueling_GANDDQN_G-v2.dump'
        fname_D_model = './saved_agents/Dueling_GANDDQN/Dueling_GANDDQN_D-v2.dump'

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

    def prep_minibatch(self):
        transitions= self.memory.random_sample(self.batch_size)

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
        value_particles, advantages = self.G_target_model(next_states, noise)
        value_means = value_particles.mean(1)  # 1 x batch_size
        action_values = value_means.view(self.batch_size, 1).expand(self.batch_size, self.num_actions) + advantages
        return action_values.max(1)[1]

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size()).to(self.device)
        interpolates = alpha * real_data.data + (1 - alpha) * fake_data.data
        interpolates.requires_grad = True

        disc_interpolates = self.D_model(interpolates)
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

    def adjust_mse_lr(self, epoch):
        lr = self.lr_mse * (0.1 ** (epoch // 3000))
        for param_group in self.G_optimizer_MSE.param_groups:
            param_group['lr'] = lr
    
    def MSE(self, x):
        return torch.mean(0.5 * x.pow(2))

    def update(self, frame=0):    
        # self.append_to_replay(s, a, r, s_)
        if frame < self.learn_start:
            return None
        if frame % self.update_freq != 0:
            return None
        # if self.memory.__len__() != self.experience_replay_size:
        #     return None  
        print('Training.........')
        self.adjust_G_lr(frame)
        self.adjust_D_lr(frame)
        self.adjust_mse_lr(frame)

        self.G_model.eval()
        for _ in range(self.n_critic):
            # update Discriminator
            batch_vars = self.prep_minibatch()
            batch_state, batch_action, batch_reward, batch_next_state = batch_vars
            G_noise = (torch.rand(self.batch_size, self.noise_size)).to(self.device)

            # estimate
            # current V particles are used for D networks
            current_value_particles, current_advantages = self.G_model(batch_state, G_noise) 
            # current_q_values = current_value_particles.mean(1) + current_advantages.gather(1, batch_action.unsqueeze(dim=-1)).squeeze(1)
            
            # target
            with torch.no_grad():
                expected_value_particles, expected_advantages = self.G_target_model(batch_next_state, G_noise)
                # add reward
                expected_value_particles = batch_reward * 0 + 1 + self.gamma * expected_value_particles

                # get expected q value
                # value_means = expected_value_particles.mean(1)  # 1 x batch_size
                # action_values = value_means.view(self.batch_size, 1).expand(self.batch_size, self.num_actions) + expected_advantages
                # expected_q_value = action_values.max(1)[0]

            # WGAN
            self.D_model.zero_grad()
            D_real = self.D_model(expected_value_particles)
            D_real_loss = torch.mean(D_real)

            D_fake = self.D_model(current_value_particles)
            D_fake_loss = torch.mean(D_fake)

            gradient_penalty = self.calc_gradient_penalty(expected_value_particles, current_value_particles)
            
            D_loss = D_fake_loss - D_real_loss + gradient_penalty

            D_loss.backward()
            self.D_optimizer.step()

        # update G network
        self.G_model.train()
        self.G_model.zero_grad()

        batch_vars = self.prep_minibatch()
        batch_state, batch_action, batch_reward, batch_next_state = batch_vars
        # estimate particles
        current_value_particles, current_advantages = self.G_model(batch_state, G_noise) 
        D_fake = self.D_model(current_value_particles)
        G_loss_GAN = -torch.mean(D_fake)
        G_loss_GAN.backward()
        self.G_optimizer.step()

        # if frame % 3 == 0:
        self.G_model.zero_grad()
        batch_vars = self.prep_minibatch()
        batch_state, batch_action, batch_reward, batch_next_state = batch_vars
        # estimate particles
        current_value_particles, current_advantages = self.G_model(batch_state, G_noise) 
        # target particles
        expected_value_particles, expected_advantages = self.G_target_model(batch_next_state, G_noise)
        # get estimated q value
        current_value_particles_mean =  current_value_particles.mean(1)
        current_advantages_specific = current_advantages.gather(1, batch_action).squeeze(1)
        current_q_value = current_value_particles_mean + current_advantages_specific
        # get expected q value
        value_means = expected_value_particles.mean(1)  # 1 x batch_size
        action_values = value_means.view(self.batch_size, 1).expand(self.batch_size, self.num_actions) + expected_advantages
        max_action_value = action_values.max(1)[0]
        expected_q_value = self.gamma * max_action_value + batch_reward.squeeze(1)

        MSE_loss = self.MSE(current_q_value-expected_q_value)
        MSE_loss.backward()
        self.G_optimizer_MSE.step()
        # G_loss = G_loss_GAN + MSE_loss
 
        
        # for param in self.G_model.parameters():
        #     param.grad.data.clamp_(-1, 1)


        # self.train_hist['G_loss'].append(G_loss.item())
        # self.train_hist['D_loss'].append(D_loss.item())

        self.update_target_model()

        print('current q value', current_q_value)
        print('expected q value', expected_q_value)
        print('MSE loss', MSE_loss)

    
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

def calc_reward(qoe, se, threshold):
    utility = np.matmul(qoe_weight, qoe.reshape((3, 1))) + se_weight * se
    if threshold > 11:
        threshold = 11
    if utility < threshold:
        reward = 0
    else:
        reward = 1
    return utility, reward

def get_action(model, s, z, eps, device):
    if np.random.random() >= eps:
        X = torch.tensor(s).to(torch.float).to(device)
        value_particles, advantages = model.G_model(X, z)
        value_mean = value_particles.mean()  # 1 x batch_size
        action_values = value_mean + advantages  
        # print(action_values.size())
        a = action_values.max(1)[1]
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'
total_timesteps = 10000

epsilon_start    = 1.0
epsilon_final    = 0.01
epsilon_decay    = 3000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# parameters of celluar environment
ser_cat_vec = ['volte', 'embb_general', 'urllc']
band_whole_no = 10 * 10**6
band_per = 1 * 10**6
qoe_weight = [1, 4, 6]
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

model = WGAN_GP_Agent(device=device, state_size=3, noise_size=10, num_actions=num_actions, num_particles=30)
# model.load_w()
G_noise = (torch.rand(1, model.noise_size)).to(torch.device(device))
env.countReset()
env.activity()
observation = state_update(env.tx_pkt_no, env.ser_cat)
print(observation)
# observation = torch.from_numpy(observation).unsqueeze(0).to(torch.float)

log = {}
rewards = []
# utility = [0.]
observations = []
actions = []
SE = []
QoE = []

for t in range(1, total_timesteps + 1):
    # env.render()
    # epsilon = exploration.value(t)
    epsilon = epsilon_by_frame(t)
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
    threshold = 6 + 4 * t / (total_timesteps/1.5) 
    print(threshold)
    utility, reward = calc_reward(qoe, se, threshold)
    QoE.append(qoe.tolist())
    SE.append(se[0])
    rewards.append(reward)
    # kpis.append(kpi)

    observation = state_update(env.tx_pkt_no, env.ser_cat)
    print(observation)
    # observation = torch.from_numpy(observation).unsqueeze(0).to(torch.float)

    model.append_to_replay(prev_observation, action, reward, observation)
    model.update(t)
    env.countReset()
    env.activity()
    print('DuelingGANDDQN=====episode: %d, epsilon: %.3f, utility: %.5f, reward: %.5f' % (t, epsilon, utility, reward))
    print('qoe', qoe)
    print('bandwidth-allocation solution', action_space[action])

    # plot_rewards(rewards)

    if t % 200 == 0:
        print('frame index [%d], epsilon [%.4f]' % (t, epsilon))
        model.save_w()
        log['state'] = observations
        log['action'] = actions
        log['SE'] = SE
        log['QoE'] = QoE
        log['reward'] = rewards

        f = open('./log/Dueling_GANDDQN/log_10M_1M_LURLLC_10.txt', 'w')
        f.write(json.dumps(log))
        f.close()
    
print('Complete')
plt.ioff()
plt.show() 

