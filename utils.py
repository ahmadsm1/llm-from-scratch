import torch
import torch.nn as nn

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4*config["emb_dim"]),
            GELU(),
            nn.Linear(4*config["emb_dim"], config["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads

        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )
        
    def forward(self, x):
        n_corpus, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(n_corpus, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(n_corpus, num_tokens, self.num_heads, self.head_dim)
        values = values.view(n_corpus, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attention_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vecs = (attention_weights @ values).transpose(1,2)

        context_vecs = context_vecs.contiguous().view(n_corpus, num_tokens, self.d_out)
        context_vecs = self.out_proj(context_vecs)    # not necessary but commonly used in llms

        return context_vecs
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, result):
        mean = result.mean(dim=-1, keepdim=True)
        variance = result.var(dim=-1, keepdim=True, unbiased = False)
        normalized_result = (result-mean) / torch.sqrt(variance+self.eps)
        return self.scale*normalized_result + self.shift

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            dropout=config["drop_rate"], 
            num_heads=config["n_heads"], 
            qkv_bias = False
        )
        self.ff = FeedForward(config)
        self.ln1 = LayerNorm(config["emb_dim"])
        self.ln2 = LayerNorm(config["emb_dim"])
        self.drop_residual = nn.Dropout(config["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.drop_residual(self.attention(self.ln1(x)))
        x = x + shortcut

        shortcut = x
        x = self.drop_residual(self.ff(self.ln2(x)))
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embdgs = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embdgs = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_embdgs = nn.Dropout(config["drop_rate"])
        
        # Transformer block placeholder
        self.transformer_block = nn.Sequential(*[TransformerBlock(config) for _ in range (config["n_layers"])])

        # LayerNorm placeholder
        self.normalization_layer = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, sequence_len = in_idx.shape
        tok_embdgs = self.tok_embdgs(in_idx)
        pos_embdgs = self.pos_embdgs(torch.arange(sequence_len, device=in_idx.device))
        return self.out_head(self.transformer_block(self.drop_embdgs(tok_embdgs + pos_embdgs)))