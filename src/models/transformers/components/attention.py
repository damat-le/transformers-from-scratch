import torch
import torch.nn as nn


class CausalAttention(nn.Module):

    def __init__(self, 
        emb_dim: int, 
        context_len: int, 
        dropout_rate: float =0.2
    ):
        
        super().__init__()

        self.emb_dim = emb_dim
        self.context_len = context_len

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Mask out the upper triangular part of the attention score matrix.
        # This is to ensure that the model does not attend to future tokens.
        # @NOTE: we use buffer to automatically move the tensor to the same device of the model.
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, X_emb: torch.Tensor) -> torch.Tensor:

        # The input embeddings have shape (batch_size, context_len, emb_dim)
        assert X_emb.shape[-1] == self.emb_dim
        assert X_emb.shape[-2] == self.context_len

        # Project the input embeddings into query, key, and value
        X_q = self.W_q(X_emb)
        X_k = self.W_k(X_emb)
        X_v = self.W_v(X_emb)

        # Compute Q•K^T 
        # (mT transposes the last two dimensions, thus accounting for minibatches)
        att_score = X_q @ X_k.mT

        # Fill the masked part with negative infinity.
        # This will ensure that the softmax will be zero in the masked part and the model will not attend to the masked tokens.
        # @NOTE: The trailing underscore means in-place operation
        att_score.masked_fill_(self.mask.bool(), float('-inf'))

        # Q•K^T / √(emb_dim)
        # Normalize the attention scores by the square root of the dimension of the key vectors.
        # @NOTE: this allows to avoid small gradients and improve the training stability
        att_score_norm = att_score / (X_k.shape[-1] ** 0.5)

        # softmax(Q•K^T / √(emb_dim))
        # Compute the attention weights
        # The weights along the rows will sum up to 1  
        att_w = torch.softmax(att_score_norm, dim=-1)

        # Apply dropout to the attention weights.
        # @NOTE: When applying the dropout with rate 0.5, to compensate for the reduction in active elements, the values of the remaining elements in the matrix are scaled up by a factor of 1/0.5 = 2.
        att_w = self.dropout(att_w)

        # softmax(Q•K^T / √(emb_dim)) • V
        # Compute the weighted sum of the value vectors
        X_context = att_w @ X_v

        return X_context