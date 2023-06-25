
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
    def __init__(self, activation, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed,  symmetry=True):
        super(NeuralNet_ev, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension    # can extend to approach to higher dimensions 
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed
        
        self.symmetry = symmetry # learn symmetric function 
        self.activation = activation
        
        self.ev_in = nn.Linear(1,1)     # eigenvalue transformation 
        self.input_layer = nn.Linear(self.input_dimension + 1 ,self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.res_layer = nn.Linear(self.neurons+1, self.neurons)    # They feed in eigenvalues right before output layer
        self.output = nn.Linear(self.neurons+1, 1)
        
        #self.init_xavier()  # Problem is that xavier gives right now recommend initialization for tanh activation, for sin not implemented. Perhaps it will be better to stick with default initialization 
        # or try out different things. Or we stick with tanh activation. In general it is kind of sketchy approach to proof effectiveness of your numerical method
        # on an analytical solvable problem, if you feed it to much knowledge about the analytical solution
        
    def forward(self, x):
        # changed your forward function slightly and added res_layer. the shapes before did not match and gave errors. Also think they
        # had a res layer just before output. so in hidden layers think is better if just put in x and x_neg otherwise if concat. with 
        # eigenvalues a start but don't feed them in with every layer then looses its meaning...
        eigenvalue = self.ev_in(torch.ones_like(x))

        x_neg = self.input_layer(torch.cat((-1*x,eigenvalue), 1))  #negated input for symmetry transformation
        x = self.input_layer(torch.cat((x,eigenvalue), 1))

        x_neg = self.activation(x_neg)
        x = self.activation(x)

        for k, l in enumerate(self.hidden_layers):
            x_neg = self.activation(l(x_neg))
            x = self.activation(l(x))

        x_neg = self.res_layer(torch.cat((x_neg,eigenvalue), 1))
        x = self.res_layer(torch.cat((x,eigenvalue), 1))
        x_neg = self.activation(x_neg)
        x = self.activation(x)

        if self.symmetry:
            out = self.output(torch.cat((x + x_neg, eigenvalue), 1))
        else:
            out = self.output(torch.cat((x - x_neg, eigenvalue), 1))
            
        return out, eigenvalue
    
    """def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)"""
    
 
        
class ev_pinn(nn.Module):

    def __init__(self, neurons, xL, xR, grid_resol, batchsize, retrain_seed):
        super(ev_pinn, self).__init__()

        self.domain_extrema = torch.tensor([xL, xR])
        self.activation=  nn.Tanh()      # or  sin_wrapper()

        """self.solution = NeuralNet_ev(self.activation,input_dimension=self.domain_extrema.shape[0], output_dimension=1,
                                              n_hidden_layers=2,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42,
                                              
                                              symmetry=True,)"""
        
        self.solution = NeuralNet_ev(self.activation,input_dimension=1, output_dimension=1,
                                              n_hidden_layers=0,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42,
                                              
                                              symmetry=True,)
        # input_dimension=self.domain_extrema.shape[0] does not give right number: gives 2 instead of 1. also set n_hiddenl to 0 st same architecture as they have
        
        self.grid_resol = grid_resol

        # TODO: other hyperparameters...
        # TODO: How to best handle data points... 

        self.training_pts = self.add_points()   # if we do not want batched inputs then this is enough I think
        
    
    def add_points(self):
        input_pts = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], self.grid_resol)
        return input_pts.reshape(-1,1)
    
    def parametric_conversion(self, input_pts, xL, xR):
        fb = 0.0    # TODO: adapt offset if needed
        Psi,E = self.solution(input_pts)
        g = (1 - torch.exp(-(input_pts - xL)))*(1 - torch.exp(-(input_pts - xR)))

        return fb + g*Psi, E
    
    def potential_sw(self, input_pts):
        # single well potential
        l = 1
        V0 = 20
        center = (self.domain_extrema[0] + self.domain_extrema[1]).numpy()/2    # center of well

        V_np = np.heaviside(-(input_pts.detach().numpy() - center + l), 0.5) + np.heaviside(input_pts.detach().numpy() - center - l, 0.5)
        V_torch = t = torch.from_numpy(V_np)

        return V_torch
    

    def compute_loss(self, input_pts):
        input_pts.requires_grad = True
        
        xL = self.domain_extrema[0]
        xR = self.domain_extrema[1]

        V = self.potential_sw(input_pts)

        # calculate pde loss
        f, E = self.parametric_conversion(input_pts, xL, xR)
        grad_f_x = torch.autograd.grad(f.sum(), input_pts, create_graph=True)[0]    # TODO: Check shapes
        grad_f_xx = torch.autograd.grad(grad_f_x.sum(), input_pts, create_graph=True)[0]

        pde_residual = grad_f_xx/2 + (E - V)*f
        pde_loss = torch.mean(pde_residual**2)

        # calculate normal loss
        norm_loss = torch.dot(f.squeeze(),f.squeeze()) - self.grid_resol/(xR - xL)


        # TODO: Ortho loss


        return pde_loss, norm_loss
    
    def fit(self):
            
        out_pred, eigenvalue = self.solution(self.training_pts)
        pde_loss, norm_loss = self.compute_loss(self.training_pts)
    
        return out_pred, eigenvalue
    


    

if __name__ == "__main__":
    neurons = 10
    retrain_seed = 42
    batchsize = 10
    grid_resol = 100
    xL = 0
    xR = 10
    pinn = ev_pinn(neurons, xL, xR, grid_resol, batchsize, retrain_seed)
    pinn.fit()
    














### the original version of your class ###

"""class NeuralNet_ev(nn.Module):
    # Modified NeuralNet to learn eigenvalue (similar to an inverse problem) and adapted forward function to inherently learn symmetry or antisymmetry solutions. Optimal weight initialization for Sin activation not clear
    def __init__(self, activation, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed,  symmetry=True):
        super(NeuralNet_ev, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension    # can extend to approach to higher dimensions 
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed
        
        self.symmetry = symmetry # learn symmetric function 
        self.activation = activation
        
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

        self.apply(init_weights)"""
    
























