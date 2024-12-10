import torch
from torch import nn
from .tokenizer import Tokenizer

class GPTPosEmbedding(nn.Module):
    """
    Positional embedding adopted by GPT models.

    It is an absolute positional embedding that is optimized during training (instead of being fixed).
    """
    def __init__(
        self, 
        vocab_size: int, 
        context_len: int, 
        emb_dim: int
        ):

        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_len, emb_dim)
        self.register_buffer(
            'positions',
            torch.arange(0, context_len)
        )


    def forward(self, tokens: torch.Tensor):
        """
        Parameters
        ----------
        X : torch.Tensor
            A tensor of shape (batch_size, seq_length) containing the input tokens.
        """
        return self.emb(tokens) + self.pos_emb(self.positions)
