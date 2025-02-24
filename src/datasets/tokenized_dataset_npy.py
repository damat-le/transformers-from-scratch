from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import Dataset

class NumpyTokenDataset(Dataset):

    """
    It assumes that the tokens are stored in a numpy.memmap array.
    """

    def __init__(self, 
        path2tokens: str, 
        context_len: int, 
        stride: int = 1
        ):
        super().__init__()
        self.path2tokens = path2tokens
        self.context_len = context_len
        self.stride = stride

        tokens = np.memmap(self.path2tokens, dtype=np.uint16, mode='r')
        self.num_tokens = len(tokens)

    def __len__(self):
        # compute the number of possible windows of size (context_len+1) that can be extracted from the tokens with the given stride
        return (self.num_tokens - (self.context_len +1)) // self.stride + 1

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        
        if idx >= len(self):
            raise IndexError("Index out of range")
        
        start = idx * self.stride
        end = start + self.context_len + 1
        
        # NOTE: we are loading the tokens at each call, which is not efficient but avoids possible problems with memory locking in multi-threaded environments (?). 
        # TO BE CHECKED.
        tokens = np.memmap(self.path2tokens, dtype=np.uint16, mode='r')

        sel_tokens = tokens[start : end]
        inputs = torch.from_numpy(sel_tokens[:-1])
        targets = torch.from_numpy(sel_tokens[1:])

        return inputs, targets
