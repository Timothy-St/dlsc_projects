import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
    
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
    


if __name__ == "__main__":
    df = pd.read_csv('TrainingData.txt')
    input_meas = torch.tensor(df[['t', 'tf0', 'ts0']].values, dtype=torch.float)

    input_window = 20
    output_window = 34

    dataset = DatasetTask3(input_meas, input_window, output_window)

    batch_size = 16
    training_set = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # data_test = torch.arange(1, 210*3+1).reshape(-1,3) # shape: [210, 3]

    # Iterate over the dataloader to get batches of input-output pairs
    for i, [inputs, outputs] in enumerate(training_set):
        if i == 0:
            print('inputs: ', inputs)
            print('----------------')
            print('outputs: ', outputs)
        
        print('------------')
        print(i)
        print('input size: ', inputs.size())
        print('output size: ', outputs.size())
        print('------------')































