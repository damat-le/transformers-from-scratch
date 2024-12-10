import torch
import torch.nn as nn

from .mh_attention import MultiHeadAttention
from .pos_embedding import GPTPosEmbedding
from .tokenizer import Tokenizer

class TransformerBlock(nn.Module):

    def __init__(self,
        head_num: int,
        emb_dim: int,
        proj_dim: int,
        context_len: int, 
        dropout_rate: float =0.2
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
        self.ln_out = nn.LayerNorm(proj_dim)

        # feed-forward
        ff_expasion_dim = proj_dim * 4 #@TODO: this must be checked
        self.ff = nn.Sequential(
            nn.Linear(proj_dim, ff_expasion_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_expasion_dim, proj_dim)
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
        # this doesn't work if emd_dim != proj_dim
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