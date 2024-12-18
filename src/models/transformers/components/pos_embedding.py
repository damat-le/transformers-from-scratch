import torch
from torch import nn

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
        tokens : torch.Tensor
            A tensor of shape (batch_size, seq_length) containing the input tokens.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, seq_length, emb_dim) containing the embeddings of the input tokens
        """
        return self.emb(tokens) + self.pos_emb(self.positions)
