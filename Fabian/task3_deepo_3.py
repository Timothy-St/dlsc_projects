import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from Common import NeuralNet
import torch.optim as optim

class DatasetTask3(TensorDataset):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # one input should have form [[t, Tf(t), Ts(t)],...,[t+n, Tf(t+n), Ts(t+n)]] and 
    # corresponding output  [[Tf(t+n+1), Ts(t+n+1)],...,[Tf(t+m), Tf(t+m)]]
    def __init__(self, tensor_data, input_window, output_window):
        self.tensor_data = tensor_data
        self.input_window = input_window    # how many initial timesteps are put into the NN, so n in description above
        self.output_window = output_window  # To how many outputs are the inputs compared, so m in description above

    def __len__(self):
        # gives number of input and output pairs. so need to subtract (input_window + output_window -1). 
        # This is due to the fact that we slide a window of length (input_window + output_window) along the input data.
        return len(self.tensor_data) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        # idx is index of individual input-output pair
        inputs = self.tensor_data[idx:idx+self.input_window]
        outputs = self.tensor_data[idx+self.input_window:idx+self.input_window+self.output_window, 1:]
        return inputs, outputs

class deepONet():
#class deepONet(nn.Module):

    def __init__(self, in_channels, p, end_bias=False, branch_bias=False):
        super(deepONet, self).__init__

        """
        Network is divided into a branch and a trunknet.
        input size given by Tf,... at different times. p determines
        output size of trunk and branch net. end_bias is boolean telling if at end
        want to apply a bias. branch_bias boolean telling if branchnet has additional bias.
        """

        # TODO: generalize to multiple temperatures Tf and Ts

        self.in_channels = int(in_channels)
        self.p = int(p)
        self.end_bias = end_bias   
        self.branch_bias = branch_bias
        self.delta_t = 2478.06  # TODO: check if right delta_t

        self.trunknet = NeuralNet(input_dimension=1, output_dimension=self.p, 
                                  n_hidden_layers=4,neurons=20, regularization_param=0.,
                                  regularization_exp=2., retrain_seed=41)
        
        self.branchnet = NeuralNet(input_dimension=self.in_channels, output_dimension=self.p,
                                   n_hidden_layers=4,neurons=20, regularization_param=0.,
                                  regularization_exp=2., retrain_seed=41)

        # TODO: Hyperparameter tuning...
        # self.activation = torch.nn.Tanh() # in NeuralNet already tanh activation function

        self.endbias_val = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.branchbias_val = nn.Parameter(torch.zeros(1, requires_grad=True))


    def get_parameters(self):
        if self.end_bias:
            if self.branch_bias:
                return list(self.trunknet.parameters()) + list(self.branchnet.parameters()) + [self.endbias_val] + [self.branchbias_val]
            else:
                return list(self.trunknet.parameters()) + list(self.branchnet.parameters()) + [self.endbias_val]
        else:
            if self.branchbias_val:
                return list(self.trunknet.parameters()) + list(self.branchnet.parameters()) + [self.branchbias_val]
            else:
                return list(self.trunknet.parameters()) + list(self.branchnet.parameters())
        
    def forward(self, u):
        # in u have values of Temperature function at m=in_channels many sampling points
        # as well as values of sampling points in time -> get next timestep where want to evaluate temperature.
        # assume shape of u = [batch, in_channels, 2] with u[:,:,1] being Temp. and u[:,:,0] t.

        # TODO: check if want to give u in this way/shape...
        # TODO: Check if could implement residual layer somewhere!!!

        y = u[:,-1,0] + self.delta_t #should be last time value of inputs + delta
        x = u[:,:,1]  # extract Temp. values

        if self.branch_bias:
            x = self.branchnet(x) + self.branchbias_val
        else:
            x = self.branchnet(x)

        y = self.trunknet(y.reshape(-1,1))

        if self.end_bias:
            return torch.sum(x*y, 1) + self.endbias_val
            # return torch.dot(x,y) + self.endbias_val
        else:
            return torch.sum(x*y, 1)
            # return torch.dot(x,y)    

    def compute_loss(self, input, output, verbose=True):
        # TODO: try to adapt to task2 template, so write compute loss function
        # expect input to have shape [batch_size, input_window, 2 or 3]
        # and output [batch, output_window, 1 or 2] -> evt do 1 and just 2 deeponets for 2 Temps

        l = torch.nn.MSELoss()  # TODO: Check if better/faster option...
        batch_size_ = output.size()[0]
        output_window = output.size()[1]
        output_pred = torch.zeros(batch_size_, output_window, 1)
        input_clone = torch.clone(input[:,:,:2])    # copy input, here only take one temperature Tf and not Ts too
        for i in range(output_window):
            prediction = self.forward(input_clone)  # until now only one Temp.
            prediction = prediction.reshape(-1,1)
            output_pred[:,i] = prediction
            delta_tensor = input_clone[:,-1,0] + self.delta_t * torch.ones(input_clone[:, -1, 0].shape) # tensor with next timestep in it
            new_tuple = torch.stack([delta_tensor, prediction[:, 0]], dim=1)    # new predicted temp. values with corresponding time
            input_clone = torch.cat([input_clone[:,1:,:], new_tuple.unsqueeze(1)], dim=1)   # append new prediction to input and recursively predict

            # TODO: generalize such that take both temperatures...or not and take 2 nets...

        loss_don = l(output_pred, output[:,:,0].unsqueeze(2))    # until now only one Temp.

        if verbose: 
            print("Total loss: ", round(loss_don.item(), 4))
        
        return loss_don
    
    def eval(self, input, output_window):
        # given input returns next n=output_window predictions
        # here input should be size [input_window, 2 or 3]

        output_pred = torch.zeros(output_window, 1)
        input_clone = torch.clone(input[:,:2])   
        for i in range(output_window):
            prediction = self.forward(input_clone.reshape(1,-1,2))
            prediction = prediction.reshape(-1,1)
            output_pred[i] = prediction

            delta_tensor = input_clone[-1,0] + self.delta_t # tensor with next timestep in it
            # delta_tensor = input_clone[-1,0] + self.delta_t * torch.ones(input_clone[-1, 0].shape)
            new_tuple = torch.stack([delta_tensor.reshape(1,1), prediction], dim=0)    # new predicted temp. values with corresponding time
            input_clone = torch.cat([input_clone[1:,:], new_tuple.reshape(1,2)], dim=0)   # append new prediction to input and recursively predict
            # input_clone = torch.cat([input_clone[1:,:], new_tuple.unsqueeze(1)], dim=1)   # append new prediction to input and recursively predict

        return input_clone
    
    def fit(self, epochs, optimizer, freq_print, training_set):
        history = list()
        verbose = False

        for epoch in range(epochs):

            if epoch % freq_print == 0:
                print("################################ ", epoch, " ################################")
                verbose = True

            for step, (input_batch, output_batch) in enumerate(training_set):
                def closure():
                    optimizer.zero_grad()
                    loss = don.compute_loss(input_batch, output_batch, verbose=verbose)
                    loss.backward()
                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)
            verbose = False
        print('Final Loss: ', history[-1])

        return history
    


if __name__ == "__main__":

    df = pd.read_csv('TrainingData.txt')
    input_meas = torch.tensor(df[['t', 'tf0', 'ts0']].values, dtype=torch.float)

    # dataloader parameters
    input_window = 120
    output_window = 20
    batch_size = 16

    dataset = DatasetTask3(input_meas, input_window, output_window)
    training_set = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    learning_rate = 0.001

    epochs = 1
    step_size = 50
    gamma = 0.5

    # in_channels = 1.*input_window
    p = 20

    don = deepONet(input_window, p, False, False)

    """optimizer = Adam(don.get_parameters(), lr=learning_rate, weight_decay=1e-5)"""

    optimizer = optim.LBFGS(don.get_parameters(),
                                lr=float(0.5),
                                max_iter=50000,
                                max_eval=50000,
                                history_size=150,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1.0 * np.finfo(float).eps)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    freq_print = 20
    
    history = don.fit(epochs, optimizer, freq_print, training_set)

    test_set = input_meas[-input_window:,:2]
    out_pred = don.eval(test_set, output_window)

    plt.figure()
    plt.plot(input_meas[:,0].detach(), input_meas[:,1].detach(),label='Tf')
    plt.plot(out_pred[:,0].detach(), out_pred[:,1].detach(),label='prediction')
    plt.show()

    






















