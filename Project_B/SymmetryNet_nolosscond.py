import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import copy
import logging

### Here learn symmetry -> so have no termination condition other than number of epochs (different for every sol). This code can also be used to find which
### number of epochs and loss_cond work best. Also tried step activation fct for symmetry node, but had error in deepcopy that I dont know how to solve.
### Tried possible residual connection -> does not seem to give better results, but could be useful when look at generalisations...
### Also tried to punish higher gradients slightly st learns lower eigenvalues but did not really work. 

### TODO: same stuff as in other code
###       1) still need better way to avoid learning high eigenvalues...
###       2) optimize values for n_epochs and loss_conditions further


class sin_wrapper(nn.Module):
    # method copied from their code
    @staticmethod
    def forward(input):
        return torch.sin(input)
    
class heavi_wrapper(nn.Module):
    # method copied from their code
    @staticmethod
    def forward(input):
        return torch.heaviside(input, torch.tensor([0.5]))

def closest_value(my_list, target):
    return min(my_list, key=lambda x: abs(x - target))

class SymmetrySwitchNet(nn.Module):
    # Modified NeuralNet to learn eigenvalue (similar to an inverse problem) and adapted forward function to inherently learn symmetry or antisymmetry solutions. Optimal weight initialization for Sin activation not clear
    def __init__(self, activation,domain_extrema, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed,symmetry=True):
        super(SymmetrySwitchNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension    # can extend to approach to higher dimensions 
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed
        
        self.activation = activation
        self.domain_extrema = domain_extrema
        
        self.ev_in = nn.Linear(1,1)     # eigenvalue transformation 
        # self.eigenvalue = None
        
        self.symmetry_neuron = nn.Linear(1,1) # learn if symmetric or antisymmetric
        # self.symmetry_switch = symmetry

        self.symmetry_switch = symmetry
        
        self.input_layer = nn.Linear(self.input_dimension ,self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.res_layer = nn.Linear(self.neurons+1, self.neurons)    # They feed in eigenvalues right before output layer
        self.output = nn.Linear(self.neurons, 1)
        
    def parametric_conversion(self, input_pts, NN_output):
        xL = self.domain_extrema[0]
        xR = self.domain_extrema[1]
        L = xR- xL
        
        fb = 0.0    # offset if needed
        g= torch.cos(np.pi/L  * input_pts)   #preserves symmetry 
        
        return fb + g*NN_output
     
        
    def forward(self, input_pts):
        eigenvalue = self.ev_in(torch.ones_like(input_pts))
        # self.eigenvalue = eigenvalue[0]
        # Determine wether to learn a symmetric or antisymmetric function
        
        #commend following two lines if want to go back to hard coding the symmetries
        # self.symmetry_switch = torch.heaviside(self.symmetry_neuron(torch.ones(1)), torch.tensor([0.5]))
        # self.symmetry_switch = heavi_wrapper()(self.symmetry_neuron(torch.ones(1)))       # Tried step function as activation function but gives error with deepcopy that I dont know how to solve
        switch_val = nn.Sigmoid()(self.symmetry_neuron(torch.ones(1)))       
        self.symmetry_switch = switch_val >=0.5    #learns better with sigmoid => interpretation of a probability distribution, play with threshold eg use 0.4 for bias towards symmetric
        
        x_neg = self.input_layer(-1*input_pts)  #negated input for symmetry transformation
        x = self.input_layer(input_pts)     

        x_neg = self.activation(x_neg)
        x = self.activation(x)

        for k, l in enumerate(self.hidden_layers):
            x_neg = self.activation(l(x_neg))
            x = self.activation(l(x))

        # possible residual connection -> does not seem to give better results, but could be useful when look at generalisations
        # x = self.activation(self.res_layer(torch.cat((x,eigenvalue), 1)))
        # x_neg = self.activation(self.res_layer(torch.cat((x_neg,eigenvalue), 1)))
        
        x_neg_out= self.activation(self.output(x_neg))
        x_out= self.activation(self.output(x))

        if self.symmetry_switch:
            out = x_out + x_neg_out
        else:
            out = x_out - x_neg_out
        
        out = self.parametric_conversion(input_pts,out)   
        
        return out, eigenvalue
    
          
class ev_pinn(nn.Module):

    def __init__(self, neurons, xL, xR, grid_resol, batchsize, retrain_seed, sigma):
        super(ev_pinn, self).__init__()

        self.xL = xL
        self.xR = xR
        self.domain_extrema = torch.tensor([xL, xR],dtype=torch.float32)
        self.activation=  nn.Tanh() #sin_wrapper()

        self.eigenf_list = []   # should contain all the found eigenfunctions in a list
        self.eigen_vals = []   # ToDo: still need to add eigenvals (later)
        
        self.symmetry= True

        
        self.solution = SymmetrySwitchNet(self.activation, self.domain_extrema, input_dimension=1, output_dimension=1,
                                              n_hidden_layers=0,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42,
                                              symmetry=self.symmetry)
        
        self.grid_resol = grid_resol
        self.sigma = sigma              # for randomizing points -> should be in [0,1) such that order of grid points is the same

        self.training_pts = self.add_points()   # if we do not want batched inputs then this is enough I think
        
    
    def add_points(self): 
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
        pde_residual = grad_f_xx +  E *f   #   (E - V)*f      #Small mistakes in pde_res: formerly was   grad_f_xx/2 +  E *f    think that is later for hydrogen 
        pde_loss = torch.mean(pde_residual**2)
        
        return pde_loss
    
    # TODO: evt try slight gradient punishment -> no does not help!
    
    def compute_norm_loss(self, input_pts):
        # discrete squared integral needs to equal 1
        f, E = self.solution(input_pts)
        xL = self.domain_extrema[0]
        xR = self.domain_extrema[1]
        # norm_loss = (torch.dot(f.squeeze(),f.squeeze()) - self.grid_resol/(xR - xL)).pow(2)   # pow(2) also to ensure positive values
        norm_loss = abs(torch.dot(f.squeeze(),f.squeeze()) - self.grid_resol/(xR - xL))
        return norm_loss
    
    def compute_ortho_loss_sum(self, input_pts):
        # ortho condition as in paper -> sum of all found eigenf
        psi_eigen = torch.zeros(input_pts.size())
        for NN in self.eigenf_list:
            psi_eigen += NN(input_pts)[0]
        ortho_loss = abs(torch.dot(psi_eigen.squeeze(), self.solution(input_pts)[0].squeeze()))   #use pow(2) but at least abs => ensure positive loss
        # ortho_loss = (torch.dot(psi_eigen.squeeze(), self.solution(input_pts)[0].squeeze())).pow(2)
        return  ortho_loss     
    
    def compute_ortho_loss_single(self, input_pts):
        # alternative ortho cond -> dot product with each eigenf
        res = 0
        for NN in self.eigenf_list:
            res += (torch.dot(NN(input_pts)[0].squeeze(), self.solution(input_pts)[0].squeeze())).pow(2)  #important to take absolute value or pow(2) here 
        return res/len(self.eigenf_list)       # think because sum up all residuals will have higher ortholoss for more eigenfcts but seems to be reasonable for our case
        # return res #/len(self.eigenf_list)       # if devide by len... othogonality constrain is relaxed for higher iterations
    
    def compute_loss(self, input_pts, verbose=True):
        input_pts.requires_grad = True
        xL = self.domain_extrema[0]
        xR = self.domain_extrema[1]
        f, E = self.solution(input_pts)
        
        pde_loss = self.compute_pde_loss(input_pts)
        norm_loss = self.compute_norm_loss(input_pts)


        if len(self.eigenf_list) > 0:
            # ortho_loss = self.compute_ortho_loss_sum(input_pts)         # Not observing a big difference
            ortho_loss = self.compute_ortho_loss_single(input_pts)       # suggest to focus on this one, computation not limiting
            # small_eigvals =   (self.solution.eigenvalue - max(self.eigen_vals))**2  #penalize learning high eigenvalues
            # small_eigvals = E[0]**2
            # small_eigvals = (self.solution.eigenvalue)**2
            
        else:
            ortho_loss = torch.zeros(1)
            # small_eigvals = E[0]**2
            # small_eigvals = (self.solution.eigenvalue)**2
   
        # loss = torch.log10( pde_loss + norm_loss + ortho_loss)  
        
        # reg_decay = len(self.eigenf_list)   
        # not enough time to make this consistent with optimization. Idea was to decrease the regularization with number of eigenfunctions learned, but with current L =6 probably not necessary, as eigenvalues of similar magnitude
        # loss = ( pde_loss + norm_loss + ortho_loss + 0.5*small_eigvals)    
        loss = ( pde_loss + norm_loss + ortho_loss ) 
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(pde_loss).item(), 4), "| Norm Loss: ", round(torch.log10(norm_loss).item(), 4),
                          "| Ortho loss: ",round(torch.log10(ortho_loss).item(), 4),  "| symmetry: ", self.solution.symmetry_switch )

        return loss
    
    def perturb_pts(self, input_pts, xL, xR):
        # assume input_pts to be evenly spaced as provided by training_pts
        delta_x = input_pts[1] - input_pts[0]
        noise = torch.rand_like(input_pts) # uniform in [0,1)     
        noise = (2*noise - 1)*delta_x/2*self.sigma     #noise should now be in [-delta_x/2, +delta_x/2) interval -> so order after perturbation of points shoul be conserved
        input_pts = torch.clamp(input_pts + noise, min=xL, max=xR)  # should still lie in [xL, xR] range

        # TODO: evt better for end values of input_pts to have same values xL nd xR...

        return input_pts
    
    
    def exact_solution(self,pts, n=1):
        #returns exact solution for free particle V=0:
        L = (self.xR- self.xL)
        c = np.sqrt( 2 / L)
        lam = (np.pi * n / L )
        return c * torch.sin( lam  *  (pts + L/2)), lam**2

    def plotting(self, n=1):
        pts= self.add_points()
        f, E = self.solution(pts)
        E = round(E[0].item(), 4)
        
        exact_f, excact_E = self.exact_solution(pts, n)
        excact_E = round(excact_E, 4)
        
        plt.figure()
        plt.plot(pts.detach(), f.detach(), label= f'Approximate E: {E}')
        plt.plot(pts.detach(), exact_f.detach(), label= f'Exact E: {excact_E}')
        plt.plot(pts.detach(), - exact_f.detach(), label= f'Exact E: {excact_E}')
        plt.legend()
        plt.show()

    def fit_single_function(self, optimizer, epochs, verbose=False):
        #current hyper params
        window = 100
        exp_rm_threshold = -2   #if use beta in ADAM observed that can use eg -10 
    
        history = []
        # Setup logger
 
        # Loop over epochs
        for epoch in range(epochs):
            # verbose = (epoch % 300  == 0)
            if verbose: print("################################ ", epoch, " ################################")
            
            def closure():
                optimizer.zero_grad()

                # TODO: perturb points or not?

                # loss = self.compute_loss(self.perturb_pts(self.training_pts, self.domain_extrema[0], self.domain_extrema[1]), verbose=verbose)
                loss = self.compute_loss(self.training_pts, verbose)    #ToDo: Cheack wether pertubation helps learning ==> perturbation might suboptimal for orthogonality
                loss.backward()
                history.append(loss.item())
                return loss

            optimizer.step(closure=closure)
            
            #rolling mean for patience condition
            if len(history) >= window+1:
              rm = np.mean(np.array(history[-window:])-np.array(history[-window-1:-1]))
            else:
              rm = np.mean(np.array(history[1:])-np.array(history[:-1]))

        # since have no ortho switch removed condition, but evt put back in st dont save wrong sols...
            
        self.eigenf_list.append(copy.deepcopy(self.solution))
        print('--------------------------------------')
        print(f'rm value: {rm} and loss {history[-1]}')
        print('--------------------------------------')
        self.plotting(len(self.eigenf_list))
        del self.solution
        self.solution = SymmetrySwitchNet(self.activation, self.domain_extrema, input_dimension=1, output_dimension=1,
                                              n_hidden_layers=0,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42,
                                            #   symmetry= symmetry_change
                                              )
        return history, epoch   

    def learn_eigenfunction_set(self,no_of_eingen, epochs_arr, verbose= False):
        history = []
        iterations = []

        # changed it such that give max epochs for each sol and not in total, bc if have no loss cond 
        
        for n in range(no_of_eingen):
            # optimizer = optim.Adam(self.parameters(), lr=lr)  #Suggest to focus on ADAM, better tailored for patience cond + comparable
            # for further improvement probably need to play with the betas. Observed that then best exp_rm_threshold also changes

            # inserted weight_decay here. number can be further optimized but think in this range is not bad
            optimizer = optim.Adam(self.parameters(), lr=5e-3, weight_decay=1e-2)
            # optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-3, betas=[0.999, 0.9999])
       
            history_n , epochs_needed =self.fit_single_function(optimizer,epochs_arr[n],verbose= verbose)
            history += history_n
            iterations.append(epochs_needed+1)
        return history 
  
def plot_hist(hist):
    plt.figure(dpi=150)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
    # plt.xscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    neurons = 20
    retrain_seed = 42
    batchsize = 10
    grid_resol = 100
    
    a = 3
    xL = -a; xR = a # shift with 0 into center in order to apply symmetry transformation
    sigma = 0.5
    pinn = ev_pinn(neurons, xL, xR, grid_resol, batchsize, retrain_seed , sigma)

    n_solutions = 5

    # different number of iterations for different solutions
    epochs_arr = [4000, 5000, 6000, 6000, 6000]

    # epochs = 6000
    
    lr = 1e-2
    betas = [0.999, 0.9999]
    
    history =pinn.learn_eigenfunction_set(n_solutions, epochs_arr)
    # history =pinn.learn_eigenfunction_set(n_solutions, epochs)
    
    plot_hist(history)






















































