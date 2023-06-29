#%%
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import copy

### update: Parametrization is now moved into NeuralNet_ev. Use cos function to preserve symmetry anti symmetry.
### removed eigenvalue from main neural net, now its just a single neuron. Dont see its meaning for learning the function, especially if also embedd symmetries.

### ToDos: Add orthonormal loss and implement patience condition.  
### learn ant-/ symmetryanti with a single neuron and threshold: if self.symm(1)>=1: symetric else asymmetric
### Suggestions: Think a patience condition is inherent to LBFGS and perhaps can more effectivly implement it with LBFGS
### Maybe try to train over random sets and points, but can be done later if rest is working


class sin_wrapper(nn.Module):
    # method copied from their code
    @staticmethod
    def forward(input):
        return torch.sin(input)


class NeuralNet_ev(nn.Module):
    # Modified NeuralNet to learn eigenvalue (similar to an inverse problem) and adapted forward function to inherently learn symmetry or antisymmetry solutions. Optimal weight initialization for Sin activation not clear
    def __init__(self, activation,domain_extrema, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed,  symmetry):
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
        self.domain_extrema = domain_extrema
        
        self.ev_in = nn.Linear(1,1)     # eigenvalue transformation 
        
        self.input_layer = nn.Linear(self.input_dimension ,self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.res_layer = nn.Linear(self.neurons, self.neurons)    # They feed in eigenvalues right before output layer
        self.output = nn.Linear(self.neurons, 1)
        
        #self.init_xavier()    causes trouble with symmetrization, assume that initializes a linear NN -> x + x_neg = 0
        
    def parametric_conversion(self, input_pts, NN_output):
        xL = self.domain_extrema[0]
        xR = self.domain_extrema[1]
        L = xR- xL
        
        fb = 0.0    # offset if needed
        g= torch.cos(np.pi/L  * input_pts)   #preserves symmetry 
        
        return fb + g*NN_output
     
        
    def forward(self, input_pts):
        eigenvalue = self.ev_in(torch.ones_like(input_pts))

        x_neg = self.input_layer(-1*input_pts)  #negated input for symmetry transformation
        x = self.input_layer(input_pts)

        x_neg = self.activation(x_neg)
        x = self.activation(x)

        for k, l in enumerate(self.hidden_layers):
            x_neg = self.activation(l(x_neg))
            x = self.activation(l(x))
        
        x_neg_out= self.activation(self.output(x_neg))
        x_out= self.activation(self.output(x))
        
        # x_neg_out= self.parametric_conversion(input_pts,x_neg_out)    since parametrization now even, can move it after symmetrization 
        # x_out= self.parametric_conversion(input_pts,x_out)

        if self.symmetry:
            out = x_out + x_neg_out
        else:
            out = x_out - x_neg_out
        
        out = self.parametric_conversion(input_pts,out)   
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

    def __init__(self, neurons, xL, xR, grid_resol, batchsize, retrain_seed, symmetry):
        super(ev_pinn, self).__init__()

        self.xL = xL
        self.xR = xR
        self.domain_extrema = torch.tensor([xL, xR],dtype=torch.float32)
        self.activation=  nn.Tanh() 

        self.eigenf_list = []   # should contain all the found eigenfunctions in a list

        
        self.solution = NeuralNet_ev(self.activation, self.domain_extrema, input_dimension=1, output_dimension=1,
                                              n_hidden_layers=0,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42,
                                              symmetry=symmetry)
        
        self.grid_resol = grid_resol

        # TODO: other hyperparameters...
        # TODO: How to best handle data points... 

        self.training_pts = self.add_points()   # if we do not want batched inputs then this is enough I think
        
    
    def add_points(self):   #TODO: implement random points and random training sets
        input_pts = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], self.grid_resol)
        return input_pts.reshape(-1,1)
    
    def potential_sw(self, input_pts):
        # single well potential
        l = 1
        V0 = 20
        center = (self.domain_extrema[0] + self.domain_extrema[1]).numpy()/2    # center of well

        V_np = np.heaviside(-(input_pts.detach().numpy() - center + l), 0.5) + np.heaviside(input_pts.detach().numpy() - center - l, 0.5)
        V_torch = torch.from_numpy(V_np)

        return V_torch
    
    def compute_pde_loss(self, input_pts): 
        input_pts.requires_grad = True
        xL = self.domain_extrema[0]
        xR = self.domain_extrema[1]
        # V = self.potential_sw(input_pts)    later also add potential 

        f, E = self.solution(input_pts)
        grad_f_x = torch.autograd.grad(f.sum(), input_pts, create_graph=True)[0]   
        grad_f_xx = torch.autograd.grad(grad_f_x.sum(), input_pts, create_graph=True)[0]
        
        # calculate pde loss
        pde_residual = grad_f_xx/2 +  E*f   #   (E - V)*f    
        pde_loss = torch.mean(pde_residual**2)
        
        return pde_loss
    
    
    def compute_norm_loss(self, input_pts):
        # discrete squared integral needs to equal 1
        f, E = self.solution(input_pts)
        xL = self.domain_extrema[0]
        xR = self.domain_extrema[1]
        norm_loss = (torch.dot(f.squeeze(),f.squeeze()) - self.grid_resol/(xR - xL)).pow(2) 
        return norm_loss
    
    def compute_ortho_loss_sum(self, input_pts):
        # ortho condition as in paper -> sum of all found eigenf
        psi_eigen = torch.zeros(input_pts.size())
        for NN in self.eigenf_list:
            psi_eigen += NN(input_pts)[0]
        ortho_loss = torch.dot(psi_eigen.squeeze(), self.solution(input_pts)[0].squeeze())
        return ortho_loss
    
    def compute_ortho_loss_single(self, input_pts):
        # alternative ortho cond -> dot product with each eigenf
        res = 0
        for NN in self.eigenf_list:
            res += torch.dot(NN(input_pts)[0].squeeze(), self.solution(input_pts)[0].squeeze())
        return res/len(self.eigenf_list)

    def compute_loss(self, input_pts, verbose=True):
        input_pts.requires_grad = True
        xL = self.domain_extrema[0]
        xR = self.domain_extrema[1]
        f, E = self.solution(input_pts)
        
        pde_loss = self.compute_pde_loss(input_pts)
        norm_loss = self.compute_norm_loss(input_pts)

        if len(self.eigenf_list) > 0:
            ortho_loss = self.compute_ortho_loss_sum(input_pts)         # TODO: check which ortho loss works better
            # ortho_loss = self.compute_ortho_loss_single(input_pts)
        else:
            ortho_loss = 0
        
        #force to learn solution for some n to get insights, remove in general
        n = 1
        E_sol = torch.full(E.shape, np.pi *n / (xR-xL) ) 
        specific_sol = torch.mean(abs(E-E_sol)**2)
        
        loss = torch.log10( pde_loss + norm_loss + ortho_loss + 10* specific_sol ) 
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(pde_loss).item(), 4), "| Norm Loss: ", round(torch.log10(norm_loss).item(), 4))

        return loss
    
    def perturb_pts(self, input_pts):

        # TODO: maybe useful like in their code to pertrub gridpoints...

        return 0
    
    
    def fit(self, num_epochs, optimizer, max_iter, loss_tolerance, verbose=True):
        # TODO: find evt better way for implementing if conditions and evaluating models -> have problem if data is batched

        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            patience_condition = False      # boolean if LBFGS met patience condition
            loss_condition = False          # True if loss is under certain value
            ortho_counter = 0               # to keep track of how many times switched symm -> max one switch

            while ortho_counter < 2:

                iter_counter = 0            # counts how many closure() calls LBFGS did

                # for j, ksk in enumerate(self.training_pts): 
                # took out this for loop because have no batches and it just looped over each entry of training points tensor
                    
                def closure():
                    nonlocal iter_counter
                    iter_counter += 1

                    optimizer.zero_grad()
                    loss = self.compute_loss(self.training_pts, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())

                    return loss

                optimizer.step(closure=closure)
            
                if iter_counter < math.ceil(1.25*max_iter):
                    patience_condition = True
                else:
                    patience_condition = False
                
                if 10**history[-1] <= loss_tolerance:
                    loss_condition = True
                else:
                    loss_condition = False

                if not(patience_condition and loss_condition):
                    # not both true at the same time -> change symmetry and try again
                    if ortho_counter:   # if ortho = 1 and cond for convergence false -> leave function
                        return -1       # stop iteration, did not find even or odd sol...
                    else:
                        self.solution.symmetry = not self.solution.symmetry
                        ortho_counter += 1
                else:
                    self.eigenf_list.append(copy.deepcopy(self.solution))   # store deep copy of NN at that point -> better than storing values on grid st can also use random perturbed points
                    # ortho_loss = self.compute_ortho_loss_sum(self.training_pts)   # just to check if ortho loss works
                    # ortho_loss = self.compute_ortho_loss_single(self.training_pts)   # just to check if ortho loss works
                    ortho_counter = 2   # leave while loop bc found sol

        print('Final Loss: ', history[-1])

        return history

    def exact_solution(self,pts, n=1):
        #returns exact solution for free particle V=0:
        L = (self.xR- self.xL)
        c = np.sqrt( 2 / L)
        eig_val = np.pi * n / L 
        return c * torch.sin( eig_val  *  (pts + L/2)), eig_val

    def plotting(self):
        pts= self.add_points()
        f, E = self.solution(pts)
        E = round(E[0].item(), 4)
        
        exact_f, excact_E = self.exact_solution(pts, n=1)
        excact_E = round(excact_E, 4)
        
        plt.figure()
        plt.plot(pts.detach(), f.detach(), label= f'Approximate E: {E}')
        plt.plot(pts.detach(), exact_f.detach(), label= f'Exact E: {excact_E}')
        plt.plot(pts.detach(), - exact_f.detach(), label= f'Exact E: {excact_E}')
        plt.legend()
        plt.show()
        
def plot_hist(hist):
    plt.figure(dpi=150)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
    plt.xscale("log")
    plt.legend()


#%%
if __name__ == "__main__":
    neurons = 20
    retrain_seed = 42
    batchsize = 10
    grid_resol = 100
    xL = -1   # shift with 0 into center in order to apply symmetry transformation
    xR = 1
    pinn = ev_pinn(neurons, xL, xR, grid_resol, batchsize, retrain_seed , symmetry=True)

    # fit and LBFGS parameters
    epochs = 1
    max_iter = 1000
    tolerance_grad = 1e-8 # patience condition tolerance -> not exactly the same bc LBFGS looks at gradient but should definitely behave similarly
    loss_tolerance = 1e-1   # loss condition tolerance
    # lr = 1e-2

    betas = [0.999, 0.9999]
    
    optimizer_LBFGS = optim.LBFGS(pinn.solution.parameters(),
                              lr=float(0.5),
                              max_iter=max_iter,
                              max_eval=max_iter*1.25,           # default value
                              tolerance_grad=tolerance_grad,
                              history_size=100,
                              line_search_fn="strong_wolfe")
    
    history =pinn.fit(epochs,optimizer_LBFGS, max_iter, loss_tolerance)

    if history == -1:
        print('failed')
    else:
        plot_hist(history)
        pinn.plotting()

    # history =pinn.fit(epochs,optimizer)
    # lot_hist(history)
    
    # pinn.plotting()
    
    










# %%
### Parametrizer comparing ###

"""xL=-1
xR=1
input_pts = torch.linspace(xL, xR, 100)

L= xR-xL
cos2_g = (1 - torch.cos(np.pi/L  * (input_pts - xL))**2 )
exp_g = (1 - torch.exp(-(input_pts - xL)))*(1 - torch.exp(-(input_pts - xR)))
sin_g= torch.cos(np.pi/L  * (input_pts))

pts= input_pts.detach()
plt.figure()
plt.plot(pts, cos2_g.detach(), label= f' (1 -cos2(a x))')
plt.plot(pts, exp_g.detach(), label= f' (1 - exp)(1-exp)')
plt.plot(pts, sin_g.detach(), label= f' cos(ax)')
plt.legend()
plt.show()"""
# asymmetric weighting of the conventional (1-e)(1-e) counter productive for learning task. 
# Asymmetric bias especially significant for unnormalized input

# %%
