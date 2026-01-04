import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

class MLPshallow(nn.Module): 
    def __init__(self, hidden_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_type = 'shallow' # for keeping info
        self.l1 = nn.Linear(1, self.hidden_dim)
        self.lout = nn.Linear(self.hidden_dim, 1)

        self.weight_history = []
            
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.lout(x)
        return x
    
    def save_weights(self): 
        '''
            saves the weights in an array
        '''
        with torch.no_grad():
            l = list(self.parameters())
            self.weight_history.append(np.copy(l[-1].numpy()))


class MLPdeep(nn.Module):
    def __init__(self, hidden_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net_type = 'deep' # for keeping info
        self.l1 = nn.Linear(1, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lout = nn.Linear(self.hidden_dim, 1)

        self.weight_history = []
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = self.lout(x)
        return x
    
    def save_weights(self): 
        '''
            saves the weights in an array
        '''
        with torch.no_grad():
            l = list(self.parameters())
            self.weight_history.append(np.copy(l[-1].numpy()))

    

def simple_f(x):
    return np.maximum(.3*x, 0)

def middle_f(x):
    return np.abs(x+.5) - 2 * np.maximum(x-.5, 0) - .4

def complex_f(x):
    return np.sin(10 * x) * np.cos(2 * x + 1) 



class Data(): 
    '''
        Generates batches of (labels, responses) pairs, of the form (x_i,  f(x_i) + noise).
        x_i are 1-dimensional
    '''

    def __init__(self, n=1000, xmin=-1, xmax=1, noise_level=1e-2, type='simple'): 
        self.n = n  # number of data points
        self.xmin = xmin # min feature  
        self.xmax = xmax # max feature 

        self.noise_level = noise_level # gaussian noise of variance noise_level**2
        self.type = type # define the target function

        self.inputs = torch.empty(n, 1) # all inputs in our dataset
        self.outputs = torch.empty(n, 1) # all responses        

        self.fill_data() # fill inputs and outputs

        self.pass_order = np.arange(n) # will be shuffled every time we go through the data
        self.current_position = 0 # current position in pass order. Used to generate batches.

    def true_f(self, x):
        if self.type == 'simple':
            return simple_f(x)
        if self.type == 'middle':
            return middle_f(x)
        if self.type == 'complex':
            return complex_f(x)
    
    def next_batch(self, batch_size=10):
        pos = self.current_position
        self.current_position = (self.current_position + batch_size) % self.n 

        indices = self.pass_order[pos: pos+batch_size]
        input_batch = torch.stack([self.inputs[i] for i in indices])
        output_batch = torch.stack([self.outputs[i] for i in indices])

        if pos + batch_size > self.n: 
            np.random.shuffle(self.pass_order)
        
        return input_batch, output_batch

    def fill_data(self):
        for i in range(self.n): 
            x = self.xmin + np.random.rand() * (self.xmax  - self.xmin)
            y = self.true_f(x)
            self.inputs[i] = x 
            self.outputs[i] = y + self.noise_level * np.random.normal()

    def __len__(self):
        return self.n    
    
################################################################################################################

def plot_net(net, n_points=5000, xmin=-1, xmax=1, label='Current approx'):
    with torch.no_grad():
        xs = torch.linspace(xmin, xmax, n_points).reshape(n_points, 1)
        nn_values = net(xs)
        plt.plot(xs, nn_values, label=label)


def plot_data(dataset, n_points=1000):
    with torch.no_grad():
        xs = torch.linspace(-1, 1, n_points).reshape(n_points, 1)
        true_vals = dataset.true_f(xs)
        plt.plot(xs, true_vals, linestyle='--', alpha=.2, label='True values')
        plt.scatter(dataset.inputs, dataset.outputs, marker='x', label='data')