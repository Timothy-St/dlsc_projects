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
    input_data = torch.tensor(df[['t', 'tf0']].values, dtype=torch.float)

    # input data shape [210, 3] -> get five input/output pairs!
    seq_length = 35

    dataset = DatasetTask3(input_data, seq_length)

    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for input, output in data_loader:
        # input_seq shape [batch_size, window_len, 2]
        # output_seq shape [batch_size, window_len, 1]
        print('-------')
        print(input)
        print('-------')
        print(output)
        print('-------')


