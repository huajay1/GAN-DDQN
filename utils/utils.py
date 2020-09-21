import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.1)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
    
# def update(self, frame=0):
#         if self.static_policy:
#             return None
        
#         # self.append_to_replay(s, a, r, s_)

#         if frame < self.learn_start:
#             return None

#         if frame % self.update_freq != 0:
#             return None

#         if self.memory.__len__() != self.experience_replay_size:
#             return None
        
#         print('Training.........')

#         # self.adjust_G_lr(frame)
#         # self.adjust_D_lr(frame)

#         self.memory.shuffle_memory()
#         len_memory = self.memory.__len__()
#         memory_idx = range(len_memory)
#         slicing_idx = [i for i in memory_idx[::self.batch_size]]

#         self.G_model.eval()
#         for t in range(len_memory // self.batch_size):
#             for _ in range(self.n_critic):
#                 # update Discriminator
#                 batch_vars = self.prep_minibatch(slicing_idx[t], slicing_idx[t+1])
#                 batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, idxes, weights = batch_vars
#                 batch_action = batch_action.unsqueeze(dim=-1).expand(-1, -1, self.num_samples)

#                 # estimate
#                 tau_ = torch.rand(self.batch_size, self.num_samples).to(self.device)
#                 current_q_values_samples = self.G_model(batch_state, tau_) # batch_size x (num_actions*num_samples)
#                 current_q_values_samples = current_q_values_samples.gather(1, batch_action).squeeze(1)

#                 # target
#                 with torch.no_grad():
#                     expected_q_values_samples = torch.zeros((self.batch_size, self.num_samples), device=self.device, dtype=torch.float) 
#                     non_final_t = tau_[non_final_mask]
#                     max_next_action = self.get_max_next_state_action(non_final_next_states, non_final_t)
#                     expected_q_values_samples[non_final_mask] = self.G_target_model(non_final_next_states, non_final_t).gather(1, max_next_action).squeeze(1)

#                     expected_q_values_samples = batch_reward + self.gamma * expected_q_values_samples

#                 D_noise = 0. * torch.randn(self.batch_size, self.num_samples).to(self.device)
#                 # WGAN-GP
#                 self.D_model.zero_grad()
#                 D_real = self.D_model(expected_q_values_samples, D_noise)

#                 D_fake = self.D_model(current_q_values_samples, D_noise)

#                 pdist = self.l1dist(expected_q_values_samples, current_q_values_samples).mul(2e-4)

#                 errD = self.leakyRelu(D_real - D_fake + pdist).mean()
#                 errD.backward()

#                 gradient_penalty = self.calc_gradient_penalty(expected_q_values_samples, D_noise)
#                 gradient_penalty.backward()

#                 gradD = expected_q_values_samples.grad
#                 D_loss = errD + gradient_penalty
#                 self.D_optimizer.step()

#             # update G network
#             self.G_model.train()
#             self.G_model.zero_grad()

#             # estimate
#             current_q_values_samples = self.G_model(batch_state, tau_) # batch_size x (num_actions*num_samples)
#             current_q_values_samples = current_q_values_samples.gather(1, batch_action).squeeze(1)
            
#             # WGAN-GP
#             D_fake = self.D_model(current_q_values_samples, D_noise)
#             errG = D_fake.mean()
#             errG.backward()
            
#             gradG = batch_state.grad
#             G_loss = errG
#             self.G_optimizer.step()

#             self.train_hist['G_loss'].append(D_loss.item())
#             self.train_hist['D_loss'].append(G_loss.item())

#             self.update_target_model()
#         if frame % 1000 == 0:
#             print('current q value', current_q_values_samples.mean(1))
#             print('expected q value', expected_q_values_samples.mean(1))