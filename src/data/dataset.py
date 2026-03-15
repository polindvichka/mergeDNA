import torch
from torch.utils.data import Dataset

class ExampleDNADataset(Dataset):

    def __init__(self, num_samples=1000, seq_length=512):
        self.num_samples = num_samples
        self.seq_length = seq_length
        # create random sequences with ATGC (0, 1, 2, 3)
        self.data = torch.randint(0, 4, (num_samples, seq_length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
