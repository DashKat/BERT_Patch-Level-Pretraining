# See Configuration @ https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/configuration_distilbert.py
# Most of these classes are built using https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py as a reference

from typing import Tuple, Optional
import torch
from torch import nn
from transformers import PretrainedConfig

DEBUG = True

class FFN(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        self.activation = nn.GELU()

    def forward(self, input: torch.Tensor)-> torch.Tensor:
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx=0):
        super().__init__()

        self.config = config
        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = nn.MultiheadAttention(embed_dim=config.dim, num_heads=config.n_heads, dropout=config.dropout, batch_first=True)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        self.factor = self.config.factors[layer_idx]

    def downscale(self, x, factor):
        B, L, D = x.shape
        L = L // factor
        x = x.view(B, L, factor, D).mean(dim=2)
        return x

    def upscale(self, x, factor):
        return x.repeat_interleave(factor, dim=1)

    def downscale_attn_mask(self, x, factor):
        if x is None:
            return None
        x = x[:, 0, 0, :] # 4D -> 2D mask (not a causal mask, just for BERT padding!)
        B, L = x.shape
        L = L // factor
        y = (x == 0) 
        return x.view(B, L, factor).all(dim=2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None, 
        head_mask: Optional[torch.Tensor] = None, #Unsupported
        output_attentions: bool = False, 
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        B, L, D = x.shape
        
        if DEBUG:
            if L % self.factor != 0:
                print("WARNING: Tokenizer must pad to factor of " + str(self.factor)) 
            
        
        attn_mask = self.downscale_attn_mask(attn_mask, self.factor)

        
        if self.factor > 1: # Preserve speed of original layer during benchmarking.
            x_d = self.downscale(x, self.factor) 
        else:
            x_d = x
        # Self-Attention
        sa_output = self.attention(
            query=x_d,
            key=x_d,
            value=x_d,
            key_padding_mask=attn_mask,
            need_weights=output_attentions,
        )

        sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        
        if self.factor > 1:

            sa_output = self.upscale(sa_output, self.factor)
        
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        if self.factor > 1:
            sa_output_ffn = self.downscale(sa_output, self.factor)
        else:
            sa_output_ffn = sa_output
        
        # Feed Forward Network
        ffn_output = self.ffn(sa_output_ffn)  # (bs, seq_length, dim)

        if self.factor > 1:
            ffn_output = self.upscale(ffn_output, self.factor)

        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output
