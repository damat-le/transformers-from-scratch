import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

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
        self.dropout = nn.Dropout(dropout_rate)
        
        # The output of attention head will be combined together using a linear layer.
        self.W_out = nn.Linear(proj_dim, proj_dim, bias=False)

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
        batch_size = X_emb.shape[0]

        # Project the input embeddings into query, key, and value
        # The projected query, key, and value matrices have shape (batch_size, context_len, proj_dim)
        X_q = self.W_q(X_emb)
        X_k = self.W_k(X_emb)
        X_v = self.W_v(X_emb)

        # Reshape the query, key, and value matrices to have multiple heads
        # This consists of splitting the last dimension (proj_dim) into head_num and head_dim
        # The projected query, key, and value matrices will have shape (batch_size, context_len, head_num, head_dim)
        assert X_q.shape == X_k.shape == X_v.shape
        target_shape = (batch_size, self.context_len, self.head_num, self.head_dim)
        X_q = X_q.view(*target_shape)
        X_k = X_k.view(*target_shape)
        X_v = X_v.view(*target_shape)

        # In order to compute the attention scores in parallel for each batch and head, we reshape the query, key, and value matrices to have shape (batch_size, head_num, context_len, head_dim)
        X_q = X_q.transpose(1,2)
        X_k = X_k.transpose(1,2)
        X_v = X_v.transpose(1,2)

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
        # Compute the weighted sum of the value vectors.
        # The weighted sum has shape (batch_size, head_num, context_len, head_dim)
        X_context = att_w @ X_v

        # Reshape X_context to have shape (batch_size, context_len, head_num, head_dim)
        X_context = X_context.transpose(1,2)

        # Concatenate the attention heads
        # The concatenated attention heads have shape (batch_size, context_len, proj_dim)
        # @NOTE on contiguous method: 
        # - A tensor is "contiguous" if its elements are stored in a single, contiguous block of memory. 
        # - After operations like transpose or permute, the memory layout of the tensor may change, leading to a non-contiguous tensor.  
        # - The view method reshapes a tensor without changing its underlying data. However, it can only be used on tensors with a contiguous memory layout.
        # - If you attempt to call view on a non-contiguous tensor, PyTorch will raise an error. To ensure that the memory layout supports view, you can call contiguous() to rearrange the tensor’s data into a contiguous block of memory.
        # - Calling contiguous() might involve a memory copy, which could have a performance cost. However, ensuring a contiguous layout is necessary for correctness when reshaping.
        # - Prefer reshape over view if you want PyTorch to handle contiguity automatically. Unlike view, reshape handles non-contiguous tensors by creating a new contiguous copy if needed. This could have a performance cost.
        X_context = X_context.contiguous().view(batch_size, self.context_len, self.proj_dim)

        # Combine the concatenated attention heads with a linear layer
        # The output has shape (batch_size, context_len, proj_dim)
        X_context = self.W_out(X_context)

        return X_context