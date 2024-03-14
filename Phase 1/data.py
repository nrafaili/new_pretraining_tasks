from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset
import numpy as np

class Dataset(TorchDataset):
    
    def __init__(self, dataset):
        self.seqs = dataset['seqs']
        self.lengths = [len(seq) for seq in self.seqs]

    def __len__(self):
        return len(self.seqs)
    
    def __avg__(self):
        return sum(self.lengths) / len(self.lengths)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq
    
