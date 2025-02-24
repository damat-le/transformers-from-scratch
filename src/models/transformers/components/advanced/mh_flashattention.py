import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadFlashAttention(nn.Module):

    def __init__(self, 
        head_num: int,
        emb_dim: int,
        proj_dim: int,
        context_len: int, 
        dropout_rate: float =0.2
    ):
        
        super().__init__()

        assert proj_dim % head_num == 0, \
        f'emb_dim ({emb_dim}) should be divisible by head_num ({head_num})'

        # compute head_dim

        self.dropout_rate = dropout_rate
        self.context_len = context_len
        self.emb_dim = emb_dim
        self.head_num = head_num
        self.head_dim = proj_dim // head_num
        # This is the dimension of the projected query, key, and value vectors.
        # This is usually equal to the original embedding dimension.
        self.proj_dim = proj_dim 

        self.W_q = nn.Linear(emb_dim, proj_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, proj_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, proj_dim, bias=False)
        
        # The output of attention head will be combined together using a linear layer.
        self.W_out = nn.Linear(proj_dim, emb_dim, bias=False)

    def forward(self, X_emb: torch.Tensor) -> torch.Tensor:

        # The input embeddings have shape (batch_size, seq_len, emb_dim)
        batch_size, seq_len, emb_dim = X_emb.size()
        assert emb_dim == self.emb_dim
        assert seq_len <= self.context_len, f"The input sequence length ({seq_len}) is longer than the context length ({self.context_len})."

        # Project the input embeddings into query, key, and value
        # The projected query, key, and value matrices have shape (batch_size, seq_len, proj_dim)
        X_q = self.W_q(X_emb)
        X_k = self.W_k(X_emb)
        X_v = self.W_v(X_emb)

        # Reshape the query, key, and value matrices to have multiple heads
        # This consists of splitting the last dimension (proj_dim) into head_num and head_dim
        # The projected query, key, and value matrices will have shape (batch_size, seq_len, head_num, head_dim)
        target_shape = (batch_size, seq_len, self.head_num, self.head_dim)
        X_q = X_q.view(*target_shape)
        X_k = X_k.view(*target_shape)
        X_v = X_v.view(*target_shape)

        # In order to compute the attention scores in parallel for each batch and head, we reshape the query, key, and value matrices to have shape (batch_size, head_num, seq_len, head_dim)
        X_q = X_q.transpose(1,2)
        X_k = X_k.transpose(1,2)
        X_v = X_v.transpose(1,2)

        # # We can replace the following code with the torch.nn.functional.scaled_dot_product_attention function that implements the scaled dot-product attention mechanism in a more efficient way
        # att_score = X_q @ X_k.mT
        # att_score.masked_fill_(self.mask.bool()[:seq_len, :seq_len], float('-inf'))
        # att_score_norm = att_score / (X_k.shape[-1] ** 0.5)
        # att_w = torch.softmax(att_score_norm, dim=-1)
        # att_w = self.dropout(att_w)
        # X_context = att_w @ X_v

        X_context = F.scaled_dot_product_attention(
            X_q, X_k, X_v, 
            is_causal=True,
            dropout_p=self.dropout_rate
        )

        # Reshape X_context to have shape (batch_size, seq_len, head_num, head_dim)
        X_context = X_context.transpose(1,2)

        # Concatenate the attention heads
        # The concatenated attention heads have shape (batch_size, seq_len, proj_dim)
        # @NOTE on contiguous method: 
        # - A tensor is "contiguous" if its elements are stored in a single, contiguous block of memory. 
        # - After operations like transpose or permute, the memory layout of the tensor may change, leading to a non-contiguous tensor.  
        # - The view method reshapes a tensor without changing its underlying data. However, it can only be used on tensors with a contiguous memory layout.
        # - If you attempt to call view on a non-contiguous tensor, PyTorch will raise an error. To ensure that the memory layout supports view, you can call contiguous() to rearrange the tensor’s data into a contiguous block of memory.
        # - Calling contiguous() might involve a memory copy, which could have a performance cost. However, ensuring a contiguous layout is necessary for correctness when reshaping.
        # - Prefer reshape over view if you want PyTorch to handle contiguity automatically. Unlike view, reshape handles non-contiguous tensors by creating a new contiguous copy if needed. This could have a performance cost.
        X_context = X_context.contiguous().view(batch_size, seq_len, self.proj_dim)

        # Combine the concatenated attention heads with a linear layer
        # The output has shape (batch_size, seq_len, emb_dim)
        X_context = self.W_out(X_context)

        return X_context
