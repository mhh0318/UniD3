import math
import numpy as np

import torch
import torch.nn as nn
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer import LinearAttentionTransformer
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearAttentionTransformerEmbedding(nn.Module):
    def __init__(self, dim, depth, n_blocks,
      num_timesteps, heads = 8, dim_head = None, causal = False,
      one_kv_head = False, reversible = False, ff_chunks = 1, ff_glu = False, 
      ff_dropout = 0., attn_layer_dropout = 0., attn_dropout = 0., blindspot_size = 1, 
      n_local_attn_heads = 0, local_attn_window_size = 128, return_embeddings = False, 
      receives_context = False, pkm_layers = tuple(), pkm_num_keys = 128,
       attend_axially = False, linformer_settings = None, 
       context_linformer_settings = None, embedding=None,
       image_cls=2887, text_cls=16640, text_length=128, image_length=1024):
        # assert (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the window size, to calculate number of kmeans cluster'
        super().__init__()
        # emb_dim = default(emb_dim, dim)
        max_seq_len = image_length + text_length
        self.max_seq_len = max_seq_len
        self.depth = depth
        emb_dim = dim
        self.emb_dim = emb_dim

        self.depth = depth
        self.n_blocks = n_blocks

        self.embedding = embedding
        self.text_cls = text_cls
        self.image_cls = image_cls

        # input_dim += 1 # Mask!!!!!

        # self.first = nn.Embedding(input_dim, emb_dim)

        self.text_length = text_length
        self.image_length = image_length 

        self.time_pos_emb = SinusoidalPosEmb(emb_dim, num_timesteps)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.Softplus(),
            nn.Linear(emb_dim * 4, emb_dim * n_blocks * depth)
        )

        # self.token_emb = nn.Embedding(num_tokens, emb_dim)
        # self.axial_pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape=(max_seq_len // local_attn_window_size, local_attn_window_size))

        self.transformer_blocks = torch.nn.ModuleList()
        for i in range(n_blocks):
            self.transformer_blocks.append(torch.nn.ModuleList())
            for j in range(depth):
                self.transformer_blocks[-1].append(
                    LinearAttentionTransformer(
                        dim, 1, max_seq_len, heads = heads, dim_head = dim_head,
                        causal = causal, 
                        ff_chunks = ff_chunks, ff_glu = ff_glu,
                        ff_dropout = ff_dropout,
                        attn_layer_dropout = attn_layer_dropout,
                        attn_dropout = attn_dropout, reversible = reversible,
                        blindspot_size = blindspot_size,
                        n_local_attn_heads = n_local_attn_heads,
                        local_attn_window_size = local_attn_window_size,
                        receives_context = receives_context,
                        pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys,
                        attend_axially = attend_axially,
                        linformer_settings = linformer_settings,
                        context_linformer_settings = context_linformer_settings))

        self.norm = nn.LayerNorm(dim)
        # self.out = nn.Linear(emb_dim, output_dim) if not return_embeddings else nn.Identity()
        self.to_logits_img = nn.Linear(emb_dim, image_cls)
        self.to_logits_txt = nn.Linear(emb_dim, text_cls)

    def forward(self, x, t, **kwargs):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        time_embed = t.view(x.size(0), 1, self.emb_dim, self.n_blocks, self.depth)
        # x = self.first(x)
        # x_embed_axial = x + self.axial_pos_emb(x).type(x.type())
        # # x_embed_axial_time = x_embed_axial + time_embed
        # h = torch.zeros_like(x_embed_axial)
        cont_emb = self.embedding(x)
        h = cont_emb


        for i, block in enumerate(self.transformer_blocks):
            for j, transformer in enumerate(block):
                h = transformer(h + time_embed[..., i, j])

        h = self.norm(h)
        logits_img = self.to_logits_img(h[:,:self.image_length,:])# B x (Ld+Lt) x n
        logits_txt = self.to_logits_txt(h[:,self.image_length:,:]) # B x (Ld+Lt) x n


        max_neg_value = -1e4
        logits_img = F.pad(logits_img,[0,self.text_cls],value=max_neg_value)
        logits_txt = F.pad(logits_txt,[self.image_cls,0],value=max_neg_value)
        logits = torch.cat([logits_img, logits_txt], dim=1) 

        return logits
