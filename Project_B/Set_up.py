
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


class NeuralNet_ev(nn.Module):
    # Modified NeuralNet to learn eigenvalue (similar to an inverse problem) and adapted forward function to inherently learn symmetry or antisymmetry solutions. Optimal weight initialization for Sin activation not clear
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed, symmetry=True, activation)
        super(NeuralNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension    # can extend to approach to higher dimensions 
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed
        
        self.symmetry = symmetry # learn symmetric function
        
        self.ev_in = nn.Linear(1,1)     # eigenvalue transformation 
        self.input_layer = nn.Linear(self.input_dimension + 1 ,self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output = nn.Linear(self.neurons+1, 1)
        
        self.init_xavier()  # Problem is that xavier gives right now recommend initialization for tanh activation, for sin not implemented. Perhaps it will be better to stick with default initialization 
        # or try out different things. Or we stick with tanh activation. In general it is kind of sketchy approach to proof effectiveness of your numerical method
        # on an analytical solvable problem, if you feed it to much knowledge about the analytical solution
        
    def forward(self, x):
        eigenvalue = self.ev_in(torch.ones_like(x))

        x_neg = self.input_layer(torch.cat((-1*x,eigenvalue), 1))  #negated input for symmetry transformation
        x = self.input_layer(torch.cat((x,eigenvalue), 1))
        
        x_neg = torch.cat((self.activation(x_neg),eigenvalue), 1)  #add ev to input for the hidden layers, can later also try without. 
        x = torch.cat((self.activation(x),eigenvalue), 1)

        for k, l in enumerate(self.hidden_layers):
            x_neg = self.activation(l(x_neg))
            x = self.activation(l(x))

        if self.symmetry:
            out = self.output(torch.cat((x + x_neg, eigenvalue), 1))
        else:
            out = self.output(torch.cat((x - x_neg, eigenvalue), 1))
            
        return out, eigenvalue
    
    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)
    
 
        
class ev_pinn(nn.Module):

    def __init__(self, neurons, xL, xR, grid_resol, batchsize, retrain_seed):
        super(ev_pinn, self).__init__()

        self.domain_extrema = torch.tensor([xL, xR])
        self.activation=  nn.Tanh()      # or  sin_wrapper()

        self.solution = NeuralNe_env(input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                              n_hidden_layers=2,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42,
                                              symmetry=True,
                                              self.activation)


        # TODO: other hyperparameters...
        # TODO: How to best handle data points... 


        self.training_set = self.assemble_dataset()
        #self.training_pts = self.add_points()
        
    def assemble_datasets(self):
        #TODO 
    
    def parametric_conversion(self):
        # TODO: 
    
    def compute_loss(self):
        
        # TODO: write compute loss function

        return 0


    

if __name__ == "__main__":
    neurons = 10
    retrain_seed = 42
    batchsize = 10
    grid_resol = 100
    xL = 0
    xR = 10
    pinn = ev_pinn(neurons, xL, xR, grid_resol, batchsize, retrain_seed)
    pinn.fit(10, True)
    








































