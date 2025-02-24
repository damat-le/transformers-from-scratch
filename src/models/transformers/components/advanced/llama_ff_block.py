import torch
import torch.nn as nn
import torch.nn.functional as F 

class LLaMAFFBlock(nn.Module):
    """
    The feed-forward block used in the LLMA model.
    """
    
    def __init__(self,
        emb_dim: int,
        ff_dim: int,
    ):
        super().__init__()

        self.w1 = nn.Linear(emb_dim, ff_dim)
        self.w2 = nn.Linear(ff_dim, emb_dim)
        self.w3 = nn.Linear(emb_dim, ff_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(X)) * self.w3(X))