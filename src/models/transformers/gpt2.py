import torch
import torch.nn as nn

from .components import (
    GPTPosEmbedding, 
    TransformerBlock
)

class GPT2(nn.Module):

    """
    The GPT-2 model architecture.

    The true GPT-2 model implements weight tying, which means that the output layer shares the same weights as the input embeddings. Thus the true GPT-2 model has fewer parameters than the one implemented here.
    """

    def __init__(self,
        vocab_size: int,
        pos_embedder: GPTPosEmbedding,
        n_transformer_blocks: int,
        head_num: int,
        context_len: int,
        emb_dim: int,
        proj_dim: int,
        dropout_rate: float =0.2
    ):

        super().__init__()

        self.pos_embedder = pos_embedder
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(
                head_num=head_num,
                emb_dim=emb_dim,
                proj_dim=proj_dim,
                context_len=context_len,
                dropout_rate=dropout_rate
            ) for _ in range(n_transformer_blocks)
        ])

        self.ln_final = nn.LayerNorm(emb_dim)
        
        self.out_layer = nn.Linear(emb_dim, vocab_size, bias=False)


    def forward(self, X_tokens: torch.Tensor) -> torch.Tensor:
        
        # The input is a tensor of shape (batch_size, context_len) containing the input token IDs.

        # Compute the positional embeddings
        # shape: (batch_size, context_len, emb_dim)
        X = self.pos_embedder(X_tokens)

        # Apply dropout
        X = self.dropout(X)

        # Apply the transformer blocks
        X = self.transformer_blocks(X)

        # Apply the final layer normalization
        X = self.ln_final(X)

        # Project the output of the transformer blocks into the vocabulary space
        # shape: (batch_size, context_len, vocab_size)
        X = self.out_layer(X)

        return X
