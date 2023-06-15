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
        outputs = self.tensor_data[upper_end:upper_end+self.window_len,1]
        return inputs, outputs

def normal(x):
    # normalize Tf and t columns
    Tf_column = x[:, 1]
    t_column = x[:,0]

    max_Tf = torch.max(Tf_column)
    min_Tf = torch.min(Tf_column)
    max_t = torch.max(t_column)
    min_t = torch.min(t_column)

    normalized_data = torch.zeros_like(x)

    normalized_data[:, 1] = (Tf_column - min_Tf)/(max_Tf - min_Tf)
    normalized_data[:, 0] = (t_column - min_t)/(max_t - min_t)

    return normalized_data, [max_Tf, min_Tf, max_t, min_t]

def invnorm(x, maxmins):
    # revert normalization back
    # maxmins = [max_Tf, min_Tf, max_t, min_t] from normal function
    Tf_column = x[:, 1]
    t_column = x[:, 0]

    denormalized_data = torch.zeros_like(x)

    # Normalizing Tf and Ts entries between 0 and 1
    denormalized_data[:, 1] = (maxmins[0] - maxmins[1])*Tf_column + maxmins[1]
    denormalized_data[:, 0] = (maxmins[2] - maxmins[3])*t_column + maxmins[3]

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

    def forward(self, x):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]

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
        self.linear_p = nn.Linear(2, self.width)  # input channel is 2: (t, Tf(t))

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x):
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)

        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        x = x.permute(0, 2, 1)

        x = self.linear_layer(x, self.linear_q)
        x = self.output_layer(x)
        return x
    

if __name__== "__main__":
    df = pd.read_csv('TrainingData.txt')
    input_meas_f = torch.tensor(df[['t', 'tf0']].values, dtype=torch.float)
    input_meas_s = torch.tensor(df[['t', 'ts0']].values, dtype=torch.float)

    # normalize data
    input_meas_f, maxmins_f = normal(input_meas_f)
    input_meas_s, maxmins_s = normal(input_meas_s)

    # dataset parameters
    window_len = 35
    batch_size = 1

    # define dataset for training for Tf and Ts
    dataset_f = DatasetTask3(input_meas_f, window_len)
    dataset_s = DatasetTask3(input_meas_s, window_len)
    training_set_f = torch.utils.data.DataLoader(dataset_f, batch_size=batch_size, shuffle=False)
    training_set_s = torch.utils.data.DataLoader(dataset_s, batch_size=batch_size, shuffle=False)

    learning_rate = 0.001

    epochs = 900
    freq_print = 25

    # FNO parameters
    modes = 18
    width = 256

    # FNO for Tf and Ts
    fno_f = FNO1d(modes, width)
    fno_s = FNO1d(modes, width)

    step_size = 50
    gamma = 0.5

    optimizer_f = Adam(fno_f.parameters(), lr=learning_rate, weight_decay=0)
    scheduler_f = torch.optim.lr_scheduler.StepLR(optimizer_f, step_size=step_size, gamma=gamma)

    optimizer_s = Adam(fno_s.parameters(), lr=learning_rate, weight_decay=0)
    scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=step_size, gamma=gamma)

    # train model first Tf and then Ts
    l_f = torch.nn.MSELoss()
    for epoch in range(epochs):
        train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(training_set_f):
            optimizer_f.zero_grad()
            output_pred_batch = fno_f(input_batch).squeeze(2)
            loss_f = l_f(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer_f.step()
            train_mse += loss_f.item()
        train_mse /= len(training_set_f)

        scheduler_f.step()

        if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse)

    print('----------------------')

    l_s = torch.nn.MSELoss()
    for epoch in range(epochs):
        train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(training_set_s):
            optimizer_s.zero_grad()
            output_pred_batch = fno_s(input_batch).squeeze(2)
            loss_s = l_s(output_pred_batch, output_batch)
            loss_s.backward()
            optimizer_s.step()
            train_mse += loss_s.item()
        train_mse /= len(training_set_s)

        scheduler_s.step()

        if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse)

    # evaluate model on test data. save predictions in csv file. plot results.

    df = pd.read_csv('TestingData.txt')
    testing_time_np = df['t'].to_numpy()
    testing_time = torch.tensor(df['t'].values, dtype=torch.float)

    test_set_f = input_meas_f[-window_len:,:].reshape(1,window_len,2)
    output_f = fno_f(test_set_f)

    test_set_s = input_meas_s[-window_len:,:].reshape(1,window_len,2)
    output_s = fno_s(test_set_s)

    input_meas_f = invnorm(input_meas_f, maxmins_f)
    input_meas_s = invnorm(input_meas_s, maxmins_s)

    output_f = (maxmins_f[0] - maxmins_f[1])*output_f + maxmins_f[1]
    output_s = (maxmins_s[0] - maxmins_s[1])*output_s + maxmins_s[1]

    total_time = torch.cat([input_meas_f[:,0], testing_time], 0)
    total_pred_f = torch.cat([input_meas_f[:,1], output_f.squeeze()[:-1]], 0)
    total_pred_s = torch.cat([input_meas_s[:,1], output_s.squeeze()[:-1]], 0)

    output_f_np = output_f.squeeze()[:-1].detach().numpy()
    output_s_np = output_s.squeeze()[:-1].detach().numpy()

    test_df = pd.DataFrame({'t': testing_time_np, 'tf0': output_f_np, 'ts0': output_s_np})

    test_df.to_csv('task3_epoch900_2fno_w256_new2.txt', index=False)

    plt.figure()
    plt.plot(total_time.detach(), total_pred_f.detach(), label = 'pred Tf0')
    plt.plot(input_meas_f[:,0].detach(), input_meas_f[:,1].detach(),label='meas Tf0')
    plt.plot(total_time.detach(), total_pred_s.detach(), label = 'pred Ts0')
    plt.plot(input_meas_s[:,0].detach(), input_meas_s[:,1].detach(),label='meas Ts0')
    plt.legend()
    plt.savefig('task3_epoch900_2fno_w256_new2.png')
    plt.show()