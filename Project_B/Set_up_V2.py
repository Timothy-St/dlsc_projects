import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class sin_wrapper(nn.Module):
    # method copied from their code
    @staticmethod
    def forward(input):
        return torch.sin(input)


class ev_pinn(nn.Module):

    def __init__(self, neurons, xL, xR, grid_resol, batchsize, retrain_seed):
        super(ev_pinn, self).__init__()

        self.domain_extrema = torch.tensor([xL, xR])

        # TODO: other hyperparameters...
        # TODO: How to best handle data points... 

        # initialize parameters
        self.neurons = neurons
        self.activation = sin_wrapper()   # think need wrapper function
        self.symmetry = True
        self.grid_resol = grid_resol
        self.batchsize = batchsize

        # inititalize architecture
        self.ev_in = nn.Linear(1,1)
        self.hidden_1 = nn.Linear(2,self.neurons)
        self.hidden_2 = nn.Linear(self.neurons+1, self.neurons)   # TODO: evt try (neurons+1, neurons+1) and the in output (neurons+2,1)
        self.output = nn.Linear(self.neurons+1, 1)

        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()

        self.training_set = self.assemble_dataset()
        #self.training_pts = self.add_points()
    
    def fit(self, num_epochs=10, verbose=True):
        # until now just to check if forward works as expexted
        # TODO: write fit function
        
        for in_pts, out_pts in self.training_set:
            out_pred, eigenvalue = self.forward(in_pts)

        # out, eigenvalue = self.forward(self.training_pts)
    
    def compute_loss(self):
        # TODO: write compute loss function

        return 0

    def assemble_dataset(self):
        input_pts = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], self.grid_resol)
        input_pts = input_pts.reshape(-1,1)
        output_pts = torch.zeros_like(input_pts)
        training_set = DataLoader(TensorDataset(input_pts, output_pts), batch_size=self.batchsize, shuffle=True)
        return training_set

    
    """def add_points(self):
        input_pts = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], self.grid_resol)
        return input_pts.reshape(-1,1)"""
    
    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)


    def forward(self, x):

        eigenvalue = self.ev_in(torch.ones_like(x))
        x_neg = self.hidden_1(torch.cat((-1*x,eigenvalue), 1))
        x = self.hidden_1(torch.cat((x,eigenvalue), 1))

        x_neg = self.activation(x_neg)
        x = self.activation(x)

        x_neg = self.hidden_2(torch.cat((x_neg,eigenvalue), 1))
        x = self.hidden_2(torch.cat((x,eigenvalue), 1))

        x_neg = self.activation(x_neg)
        x = self.activation(x)

        if self.symmetry:
            out = self.output(torch.cat((x + x_neg, eigenvalue), 1))
        else:
            out = self.output(torch.cat((x - x_neg, eigenvalue), 1))

        return out, eigenvalue
    

if __name__ == "__main__":
    neurons = 10
    retrain_seed = 42
    batchsize = 10
    grid_resol = 100
    xL = 0
    xR = 10
    pinn = ev_pinn(neurons, xL, xR, grid_resol, batchsize, retrain_seed)
    pinn.fit(10, True)
















































