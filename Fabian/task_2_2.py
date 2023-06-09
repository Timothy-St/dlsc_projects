import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Common import NeuralNet, MultiVariatePoly
import time
import pandas as pd
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)



class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_, constants):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_

        # Initialise constants for PDE
        self.alphf = constants[0]
        self.hf = constants[1]
        self.Th = constants[2]
        self.Tc = constants[3]
        self.T0 = constants[4]

        # Extrema of the solution domain (t,x) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[0, 8],  # Time dimension
                                            [0, 1]])  # Space dimension
        
        # TODO: check if better way to implement different phases -> evt already here divide domain...?

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = 1

        # TODO: check if this good lambda value and evt also train it or train it in task 1 and the use this as baseline

        # FF Dense NN to approximate the solution of the underlying heat equation
        self.approximate_sol = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=2,
                                              n_hidden_layers=5,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)

        # TODO: check if NN architecture is ideal

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Initial condition T_f(x,t=0)=T_0
    def initial_condition(self):
        return self.T0*torch.ones((self.n_tb,1))    #TODO check if shape better (N,1)!?

    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition().reshape(-1, 1)
        return input_tb, output_tb

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb)
        input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        output_sb_0 = torch.zeros((input_sb.shape[0], 1))
        output_sb_L = torch.zeros((input_sb.shape[0], 1))

        return torch.cat([input_sb_0, input_sb_L], 0), torch.cat([output_sb_0, output_sb_L], 0)

    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int
    
    def get_measurement_data(self):
        # read in txt file with meas.-points and meas.-values
        df = pd.read_csv('DataSolution.txt')
        input_meas = torch.tensor(df[['t', 'x']].values, dtype=torch.float)
        output_meas = torch.tensor(df[['tf']].values, dtype=torch.float)
        # return input_meas[::2,:], output_meas[::2]
        return input_meas, output_meas

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()  # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()  # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=2 * self.space_dimensions * self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        T_pred_tb = self.approximate_sol(input_tb)
        return T_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        T_pred_sb = self.approximate_sol(input_sb)
        return T_pred_sb
    
    def compute_pde_residual(self, input_int):

        # TODO: Find good way to implement different phases and different Uf!

        input_int.requires_grad = True
        
        T = self.approximate_sol(input_int)
        Tf = T[:,0]
        Ts = T[:,1]

        grad_Tf = torch.autograd.grad(Tf.sum(), input_int, create_graph=True)[0]
        grad_Tf_t = grad_Tf[:, 0]
        grad_Tf_x = grad_Tf[:, 1]
        grad_Tf_xx = torch.autograd.grad(grad_Tf_x.sum(), input_int, create_graph=True)[0][:, 1]

        # residual = torch.zeros(self.n_int)
        residual = 0.0

        # TODO: check if have right indexing for gradients below

        for i, [t, _] in enumerate(input_int):
            if 1<t<=2 or 3<t<=4 or 5<t<=6 or 7<t<=8:
                residual += (grad_Tf_t[i] - self.alphf*grad_Tf_xx[i] + self.hf*(Tf[i] - Ts[i]))**2
            elif 0<t<=1 or 4<t<=5:
                residual += (grad_Tf_t[i] + grad_Tf_x[i] - self.alphf*grad_Tf_xx[i] + self.hf*(Tf[i] - Ts[i]))**2
            else:
                residual += (grad_Tf_t[i] - grad_Tf_x[i] - self.alphf*grad_Tf_xx[i] + self.hf*(Tf[i] - Ts[i]))**2

        return residual/self.n_int
    
    def compute_sb_residual(self, input_sb):

        # TODO: Find a (better) way to account for different phases

        input_sb.requires_grad = True
        Tf_sb = self.approximate_sol(input_sb)[:,0]
        grad_Tf_sb_x = torch.autograd.grad(Tf_sb.sum(), input_sb, create_graph=True)[0][:, 1]

        residual_f = 0.0

        # TODO: check if have right indexing for gradients below and if shapes match and if selected right entries as x=0 or x=1!!!
        # TODO: Check if correctly normalize residual!?

        for i, [t, _] in enumerate(input_sb):
            x_zero = i<self.n_sb
            if 1<t<=2 or 3<t<=4 or 5<t<=6 or 7<t<=8:
                residual_f += (grad_Tf_sb_x[i])**2
                """if x_zero:
                    residual_f += abs(grad_Tf_sb_x[i])**2
                else:
                    residual_f += abs(grad_Tf_sb_x[i])**2"""
            elif 0<t<=1 or 4<t<=5:
                if x_zero:
                    residual_f += (Tf_sb[i] - self.Th)**2
                else:
                    residual_f += (grad_Tf_sb_x[i])**2
            else:
                if x_zero:
                    residual_f += (grad_Tf_sb_x[i])**2
                else:
                    residual_f += (Tf_sb[i] - self.Tc)**2

        return residual_f/self.n_sb

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        Tf_pred_tb = self.apply_initial_condition(inp_train_tb)[:,0]

        inp_train_meas, Tf_train_meas = self.get_measurement_data()
        Tf_pred_meas = self.approximate_sol(inp_train_meas)[:,0]

        # assert (Tf_pred_sb.shape[1] == u_train_sb.shape[1])
        # assert (Tf_pred_tb.shape[1] == u_train_tb.shape[1])
        # assert (Tf_pred_meas.shape[1] == Tf_train_meas.shape[1])

        # TODO: Check that right shape!

        loss_int = self.compute_pde_residual(inp_train_int)
        loss_sb = self.compute_sb_residual(inp_train_sb)
        r_tb = u_train_tb - Tf_pred_tb
        r_meas = Tf_train_meas - Tf_pred_meas

        loss_tb = torch.mean((r_tb) ** 2)
        loss_meas = torch.mean((r_meas) ** 2)

        loss_u = loss_sb + loss_tb + loss_meas

        loss = torch.log10(self.lambda_u * loss_u + loss_int)
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_int).item(), 4), "| Function Loss: ", round(torch.log10(loss_u).item(), 4))
        
        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True, verbose_red=False):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):

            if epoch%100==0: 
                if verbose_red: print("################################ ", epoch, " ################################")
    
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history
    
    ################################################################################################
    def plotting(self):
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)

        output = self.approximate_sol(inputs)
        # output = self.approximate_sol(inputs).reshape(-1, )

        # print('size input', inputs.size())
        # print('size output', output.size())


        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output[:,0].detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")
        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output[:,1].detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("F")
        axs[1].set_title("S")

        plt.show()


if __name__ == "__main__":
    # n_int = 256
    n_int = 512
    n_sb = 128
    # n_sb = 64
    n_tb = 64
    constants = np.array([0.005, 5., 4., 1., 1.])

    pinn = Pinns(n_int, n_sb, n_tb, constants)

    """input_sb_, output_sb_ = pinn.add_spatial_boundary_points()
    input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
    input_int_, output_int_ = pinn.add_interior_points()"""

    optimizer_LBFGS = optim.LBFGS(pinn.approximate_sol.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)

    optimizer_ADAM = optim.Adam(pinn.approximate_sol.parameters(),
                            lr=float(0.001))
    
    # TODO: Look at learning rate scheduler!
    
    n_epochs = 10000

    n_epochs_lbfgs = 10

    hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_ADAM, 
                verbose=False, verbose_red=True)
    
    """hist = pinn.fit(num_epochs=n_epochs_lbfgs,
                optimizer=optimizer_LBFGS, 
                verbose=True, verbose_red=False)"""
    
    pinn.plotting()

    """plt.figure(dpi=150)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
    plt.xscale("log")
    plt.legend()
    plt.show()"""















