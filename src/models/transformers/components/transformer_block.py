import torch
import torch.nn as nn
from .mh_attention import MultiHeadAttention

class TransformerBlock(nn.Module):

    def __init__(self,
        context_len: int,
        head_num: int,
        emb_dim: int,
        proj_dim: int,
        ff_dim: int,
        dropout_rate: float
    ):
        super().__init__()

        # layer normalization
        self.ln_in = nn.LayerNorm(emb_dim)

        # multi-head attention
        self.mha = MultiHeadAttention(
            head_num=head_num,
            emb_dim=emb_dim,
            proj_dim=proj_dim,
            context_len=context_len,
            dropout_rate=dropout_rate
        )

        # layer normalization
        self.ln_out = nn.LayerNorm(emb_dim)

        # feed-forward block
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_dim, emb_dim)
        )

        # dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    
    def forward(self, X_emb: torch.Tensor) -> torch.Tensor:

        # first layer normalization
        X = self.ln_in(X_emb)
        # multi-head attention
        X = self.mha(X)
        # dropout
        X = self.dropout(X)

        # skip connection
        X_skip = X_emb + X

        # second layer normalization
        X = self.ln_out(X_skip)
        # feed-forward
        X = self.ff(X)
        # dropout
        X = self.dropout(X)

        # skip connection
        X = X_skip + X

        return X