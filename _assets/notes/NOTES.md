# Transformers

<p align="center">
    <img src="./imgs/gpt-architecture.png" width=50%/>
</p>

## Positional Embeddings

After tokenization, the input sequence is converted into a sequence of token embeddings. Thus each token is mapped into a space of dimension `emb_dim`. Such mapping is learned during the training process through the `nn.Embedding` module in pyTorch.

The `nn.Embedding` module acts as a lookup table that maps each token to a vector in the embedding space. The embedding table is initialized randomly and learned during the training process. When a token is passed through the `nn.Embedding` module, it is first converted into a one-hot encoding of dimension `vocab_size` and then multiplied by the embedding matrix to obtain the token embedding. The code below shows how the `nn.Embedding` module is equivalent to a one-hot encoding followed by a matrix multiplication:

```python
tokens = torch.tensor([[1024, 32, 11]])
nn.Embedding(vocab_size, emb_dim)(tokens)

# is equivalent to

tokens = torch.tensor([[1024, 32, 11]])
onehot_tokens = F.one_hot(tokens, num_classes=vocab_size)
nn.Parameter(vocab_size, emb_dim)(onehot_tokens)
```

Thus, when the token `[1024]` is received by `nn.Embedding`, it is converted into a one-hot encoding of dimension `vocab_size` with a 1 at the index `1024` and zeros elsewhere. The one-hot encoding is then multiplied by the embedding matrix to obtain the token embedding. This multiplication is equivalent to selecting the row `1024` of the embedding matrix.

However, the same word can play different roles when appearing in different positions in the input sentence. Thus, cupturing the position of the token is crucial for the model to understand the context of the sentence. The embedding layer as described above does not capture the position of the token in the input sequence, since the same token will always be mapped to the same vector in the embedding space. 

To capture the position of the token, there exists several techniques. In the original transformer paper, the authors used a sinusoidal function to encode the position of the token. Such fuction is fixed and does not change during the training process. In GPT-like implementations, instead, a second `nn.Embedding` layer, called the positional embedding layer, is used and it maps the position of the token in the input sequence to a vector in the embedding space. Such positional vector is then added to the previous token embedding, enreaching the latter with positional information.

Such positional embedding layer is learned during the training process. It is a matrix of shape `(context_len, emb_dim)` where `context_len` is the length of the input sequence and `emb_dim` is the dimension of the token embedding. The way it works is the same as the token embedding layer, but instead of mapping the token index (from vocabulary space) to a vector in the embedding space, it maps the position of the token in the input sequence to a vector in the embedding space.

The way the positional embedding layer is produced in code is as follows:

```python
positions = torch.arange(context_len)
nn.Embedding(context_len, emb_dim)(positions)
```

## Layer Normalization

Layer normalizaiton is a technique to mitigate the vanishing gradient problem. It is similar to batch normalization, but instead of normalizing the activations across the batch, it normalizes the activations across the features. 

In the context of transformers, layer normalization is applied to the embedding of the tokens. Given a tensor of shape `(context_len, emb_dim)`, the layer normalization is applied to the `emb_dim` dimension. For each token it produces a normalized embedding, to ensure a stable training process. 

In this implementation of transformers we used the layer normalization provided by pyTorch. However, it is important to know what happens under the hood. The `nn.LayerNorm` module in pyTorch applies the following formula to normalize the activations:

$$
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
$$

where $x$ is the token embedding vector with dimension `emb_dim` and  $\epsilon$ is a small value to avoid division by zero. 

The parameters $\gamma$ and $\beta$ are learnable parameters that are learned during the training process. They have the same dimension as the input tensor $x$.

The `nn.LayerNorm` module is equivalent to the following code:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    std = x.var(-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(std + self.eps)
    return norm_x * self.gamma + self.beta
```

Note that `torch.var` computes the unbiased variance by default (divides by `emb_dim -1`). Here we set `unbiased=False` to compute the biased variance (divides by `emb_dim`) since for large embedding dimensions the difference is negligible.


## GELU Activation

The GELU activation function [[1]](https://arxiv.org/pdf/1606.08415) is used in transformers as the activation function in the feedforward neural network. It is a smooth approximation of the ReLU activation function. 

Computing GELU activation involves evaluating the cumulative distribution function $\Phi$ of a Gaussian Distribution:

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

This, at least in the initial implementation of GELU in TensorFlow, was very slow to compute. Hence, in this issue [[2]](https://github.com/pytorch/pytorch/issues/39853), an approximation of the GELU function was proposed. The approximation is the following:

$$
\text{GELU}(x) = 0.5x(1 + \text{tanh}(\sqrt{2/\pi}(x + 0.044715x^3)))
$$

GPT2 original implementation uses the approximation of GELU activation function. Today, probably, the original GELU is fast enough to be used in practice and thus there is no need to use the approximation. But this must be properly tested.

## Weight Tying

Weight tying [[1]](https://arxiv.org/pdf/1608.05859) is a technique used in transformers to reduce the number of parameters in the model. It consists of sharing the weights of the token embedding layer with the weights of the output layer.

The transformers blocks in a transformer architecture map the positional embeddings vectors into new vectors, with same dimension `emb_dim`, but enriched with context information. The output of the transformer block is then passed through a final linear layer to map the vectors back to the vocabulary space from embedding space.

In GPT architectures, the weights of the token embedding layer are shared with the weights of the output (linear) layer in order to reduce the amount of trainable parameters. This is done by ...

## Tensor float precision

Usually, model parameters are stored in 32-bit floating point precision. However, to speedup training, it is possible to use nvidia mixed precision training. 

In mixed precision training, torch automatically converts some of the model parameters to lower precision. 

For example, when using TF32 (32-bit TensorFloat) precision, the model parameters are stored in 32-bit floating point precision, but the matrix multiplications are performed in 32-bit TensorFloat precision. 

<p align="center">
    <img src="./imgs/fp_vs_tf.jpeg" width=50%/>
</p>

When using BF16 (16-bit BFloat) precision, some of the model parameters are stored in 16-bit bfloating point precision. The best way to use mixed precision training is to use the `torch.autocast` context manager that automatically converts what need to be converted to lower precision.

Here is a list of the cuda operations that support mixed precision training:

<p align="center">
    <img src="./imgs/cuda_cast_ops.png" width=50%/>
</p>


To have an idea of the possible speedups when redicing precision, we can use nvidia GPU specifications. For example, the nvidia A100 GPU has the following specifications:

<p align="center">
    <img src="./imgs/nvidia_a100_spec.png" width=50%/>
</p>

## Model compilation

During training, most of the time is spent in moving tensors from GPU memory to GPU cores for processing. 

By compiling the model, we can reduce the time spent in moving tensors from memory to cores. Assume you have to perform 10 operations in sequence on a tensor. By default, the tensor is moved from memory to cores and back 10 times. However, if you compile the model, the tensor is moved from memory to cores only once and the 10 operations are performed together in the cores (using cude kernel fusion).

It is possible to compile a model using `torch.compile`

Note: Flash attention is a technique to reduce the time spent in moving tensors from memory to cores specific for  attention computation. 


## Power of 2 hyperparameters

Because cuda kernels operates on block tiles (kind of data chuncks that have size of a power of 2), the performance can be increased by using powers of 2 when selecting hyperparameters of the model. In fact, when an operation must be performed on a chuck whose size is not a power of 2, the operation is done on the smallest closest power of 2 part in an optimised way and the rest of the chuck is processed by some boundary kernels in a different way. By providing chuck sizes that are powers of 2, the operations can be done in a more efficient way.

For example, by increasing the `vocab_size` to the closest power of 2, i.e. `50304`, we can have performance gains up to 5%.

Note that by increasing `vocab_size`, the model will have more parameters (~ +35k params) and the training should be in principle slower. However, the performance is still better than the previous one.
