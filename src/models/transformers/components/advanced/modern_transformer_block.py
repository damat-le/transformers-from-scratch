import torch
import torch.nn as nn
import torch.nn.functional as F
from .mh_flashattention_rope import MultiHeadFlashAttentionRope
from .llama_ff_block import LLaMAFFBlock

class ModernTransformerBlock(nn.Module):
    """
    With respect to the original GPT implementation, the following changes have been made:
    - The layer normalization has been replaced with root mean square normalization (https://arxiv.org/abs/1910.07467).
    - The multi-head attention mechanism has been replaced with the Flash attention mechanism (https://arxiv.org/abs/2205.14135).
    - Use rotary positional embeddings (https://arxiv.org/abs/2104.09864).
    """

    def __init__(self,
        context_len: int,
        head_num: int,
        emb_dim: int,
        proj_dim: int,
        ff_dim: int,
        dropout_rate: float
    ):
        super().__init__()

        # normalization
        self.rmsn_in = nn.RMSNorm(emb_dim)

        # multi-head attention
        self.mha = MultiHeadFlashAttentionRope(
            head_num=head_num,
            emb_dim=emb_dim,
            proj_dim=proj_dim,
            context_len=context_len,
            dropout_rate=dropout_rate
        )

        # normalization
        self.rmsn_out = nn.RMSNorm(emb_dim)

        # LLMA feed-forward block
        self.ff = LLaMAFFBlock(
            emb_dim=emb_dim,
            ff_dim=ff_dim
        )
        
        # dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    
    def forward(self, X_emb: torch.Tensor) -> torch.Tensor:

        # first layer normalization
        X = self.rmsn_in(X_emb)
        # multi-head attention
        X = self.mha(X)
        # dropout
        X = self.dropout(X)

        # skip connection
        X_skip = X_emb + X

        # second layer normalization
        X = self.rmsn_out(X_skip)
        # feed-forward
        X = self.ff(X)
        # dropout
        X = self.dropout(X)

        # skip connection
        X = X_skip + X

        return X