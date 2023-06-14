import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd


class DatasetTask3(TensorDataset):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # tensor_data has shape [210,3]. divide it into 5 input/output pairs of each length=window_len
    def __init__(self, tensor_data, window_len):
        self.tensor_data = tensor_data
        self.window_len = window_len    # size of input/output sequences

    def __len__(self):
        # gives number of input and output pairs. In our case it is 5.
        return int(len(self.tensor_data)/self.window_len) - 1
        # return (len(self.data) // self.seq_length) - 1

    def __getitem__(self, idx):
        # idx is index of individual input-output pair
        lower_end = idx*self.window_len
        upper_end = (idx + 1)*self.window_len
        inputs = self.tensor_data[lower_end:upper_end]
        outputs = self.tensor_data[upper_end:upper_end+self.window_len,1:]
        return inputs, outputs

def normal(x):
    # normalize Tf and Ts columns as well as t column
    Tf_column = x[:, 1]
    Ts_column = x[:, 2]
    t_column = x[:,0]

    max_Tf = torch.max(Tf_column)
    min_Tf = torch.min(Tf_column)
    max_Ts = torch.max(Ts_column)
    min_Ts = torch.min(Ts_column)
    max_t = torch.max(t_column)
    min_t = torch.min(t_column)

    normalized_data = torch.zeros_like(x)

    normalized_data[:, 1] = (Tf_column - min_Tf)/(max_Tf - min_Tf)
    normalized_data[:, 2] = (Ts_column - min_Ts)/(max_Ts - min_Ts)
    normalized_data[:, 0] = (t_column - min_t)/(max_t - min_t)

    # return normalized_data, [max_Tf, min_Tf, max_t, min_t]
    return normalized_data, [max_Tf, min_Tf, max_Ts, min_Ts, max_t, min_t]


def invnorm(x, maxmins):
    # revert normalization back
    # maxmins = [max_Tf, min_Tf, max_Ts, min_Ts] from normal function

    Tf_column = x[:, 1]
    Ts_column = x[:, 2]
    t_column = x[:, 0]

    denormalized_data = torch.zeros_like(x)

    # Normalizing Tf and Ts entries between 0 and 1
    denormalized_data[:, 1] = (maxmins[0] - maxmins[1])*Tf_column + maxmins[1]
    denormalized_data[:, 2] = (maxmins[2] - maxmins[3])*Ts_column + maxmins[3]
    # denormalized_data[:, 0] = (maxmins[2] - maxmins[3])*t_column + maxmins[3]
    denormalized_data[:, 0] = (maxmins[4] - maxmins[5])*t_column + maxmins[5]

    return denormalized_data

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)


######################### TO DO ####################################


    def forward(self, x):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]
        # hint: use torch.fft library torch.fft.rfft
        # use DFT to approximate the fourier transform
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


####################################################################


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 1  # pad the domain if input is non-periodic
        self.linear_p = nn.Linear(3, self.width)  # input channel is 2: (u0(x), x)

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 2)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        # x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)

        x = self.linear_layer(x, self.linear_q)
        x = self.output_layer(x)
        return x

"""
outline idea: Take as input [t, Tf(t), Ts(t)] or [t, Tf(t)] for n timesteps. Then use these as input and 
predict [t, Tf(t), Ts(t)] or [t, Tf(t)] for t + (n+1)*delta_t. slide window over one step and go on until have 
m new predictions. look at cummulative error over all those m predictions compared to real output. So
combine translation equiv. of FNO with approache I used for DeepONets. Also here t is input too.
"""


if __name__ == "__main__":
    df = pd.read_csv('TrainingData.txt')
    input_meas = torch.tensor(df[['t', 'tf0', 'ts0']].values, dtype=torch.float)

    # normalize data
    input_meas, maxmins = normal(input_meas)

    window_len = 35
    batch_size = 1

    dataset = DatasetTask3(input_meas, window_len)
    training_set = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    learning_rate = 0.001

    epochs = 1000
    freq_print = 100

    modes = 18
    width = 256

    fno = FNO1d(modes, width)

    step_size = 50
    gamma = 0.5

    # TODO: make such that only [t,Tf] and [t,Ts] and not both but can try both

    optimizer = Adam(fno.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    l_f = torch.nn.MSELoss()
    # history_f = list()
    for epoch in range(epochs):
        train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(training_set):
            optimizer.zero_grad()
            output_pred_batch = fno(input_batch).squeeze(2)
            loss_f = l_f(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer.step()
            # history_f.append(loss_f.item())
            train_mse += loss_f.item()
        train_mse /= len(training_set)

        scheduler.step()

        if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse)
    
    """plt.figure(dpi=150)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(history_f) + 1), history_f, label="Train Loss")
    plt.xscale("log")
    plt.legend()
    plt.show()"""

    df = pd.read_csv('TestingData.txt')
    testing_time_np = df['t'].to_numpy()
    testing_time = torch.tensor(df['t'].values, dtype=torch.float)

    test_set = input_meas[-window_len:,:].reshape(1,window_len,3)
    output = fno(test_set)

    input_meas = invnorm(input_meas, maxmins)
    output[:,:, 0] = (maxmins[0] - maxmins[1])*output[:,:, 0] + maxmins[1]
    output[:,:, 1] = (maxmins[2] - maxmins[3])*output[:,:, 1] + maxmins[3]
   
    total_time = torch.cat([input_meas[:,0], testing_time], 0)
    total_pred = torch.cat([input_meas[:,1:], output.squeeze()[:-1]], 0)

    # TODO: denormalize data!!!
    # TODO: include solid temp too -> try both inputs into fno <- next step
    # TODO: write function st get output in given csv file...
    # TODO: make it possible to run lbfgs too
    # TODO: look at difference btw shuffle=True/False <- next step
    # TODO: Look at effect of gamma <- next step
    # TODO: adjust width such that better results
    # TODO: try different optimizers and include evt L2 punishment, i.e. weight decay -> seems to be best for it to be 0...

    # testing_time_np = testing_time.detach().numpy()
    output_np = output.squeeze()[:-1].detach().numpy()

    test_df = pd.DataFrame({'t': testing_time_np, 'tf0': output_np[:,0], 'ts0': output_np[:,1]})
    test_df.to_csv('test_output_1fno_1000epoch_w256.txt', index=False)

    plt.figure()
    plt.plot(total_time.detach(), total_pred[:,0].detach(), label = 'pred tf0')
    plt.plot(total_time.detach(), total_pred[:,1].detach(), label = 'pred ts0')
    plt.plot(input_meas[:,0].detach(), input_meas[:,1].detach(),label='meas f')
    plt.plot(input_meas[:,0].detach(), input_meas[:,2].detach(),label='meas s')
    plt.legend()
    plt.show()























































