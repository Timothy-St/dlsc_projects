import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class DatasetTask3(Dataset):
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

if __name__ == "__main__":

    df = pd.read_csv('TrainingData.txt')
    # input_data = torch.tensor(df[['t', 'tf0', 'ts0']].values, dtype=torch.float)
    input_data = torch.tensor(df[['t', 'tf0']].values, dtype=torch.float)

    # Assuming your input data is a numpy array of shape [210, 3]
    seq_length = 35

    # Create an instance of the custom dataset
    dataset = DatasetTask3(input_data, seq_length)

    # Create a data loader
    batch_size = 1  # Set the batch size according to your needs
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Access the data loader
    for input_seq, output_seq in data_loader:
        # input_seq has shape [batch_size, seq_length, num_features]
        # output_seq has shape [batch_size, seq_length, num_features]
        print('-------')
        print(input_seq)
        print('-------')
        print(output_seq)
        print('-------')


