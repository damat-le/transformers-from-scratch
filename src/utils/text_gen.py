import torch
import torch.nn.functional as F

def simple_text_gen(
        seed_text,
        model,
        tokenizer,
        topk=50,
        max_len_return_seq=25,
        n_return_seq=1,
        seed = 42
    ):
    tokens = tokenizer.encode(seed_text)
    tokens = torch.tensor(
        tokens, 
        dtype=torch.long, 
        device=next(model.parameters()).device
    )
    tokens = tokens.unsqueeze(0).repeat(n_return_seq, 1)

    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)

    with torch.no_grad():
        while tokens.size(1) < max_len_return_seq:
            logits = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_idx = torch.topk(probs, topk, dim=-1)
            idx = torch.multinomial(topk_probs, 1)
            next_token = topk_idx.gather(-1, idx)
            tokens = torch.cat([tokens, next_token], dim=1)
    
    out_ = []
    for i in range(n_return_seq):
        seq = tokenizer.decode(tokens[i].tolist())
        out_.append('> ' + seq)

    return out_