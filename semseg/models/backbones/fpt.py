import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn import Dropout
from einops import rearrange, repeat
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
from operator import mul
from timm.models import VisionTransformer
from timm.layers import PatchEmbed, Mlp, DropPath, use_fused_attn
from torch.jit import Final

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3

    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, 'b (h w) d -> b d h w', h=h, w=w, d=embed_dim)
    return pos_emb

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchedInputAdapter(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int,int]],
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 learnable_pos_emb: bool = False,# change by ruiping
                #  learnable_pos_emb: bool = True,
                 image_size: Union[int, Tuple[int]] = 224):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size_full) * (self.image_size[1] // patch_size_full)

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.proj = nn.Conv2d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W)
        )

    # @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        :param x: Input image tensor
        """
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.P_H == 0) and (W % self.P_W == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}'
        N_H, N_W = H // self.P_H, W // self.P_W # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bicubic', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nh nw -> b (nh nw) d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class CrossAttention(nn.Module):
    def __init__(self, dim, fft):
        super(CrossAttention, self).__init__()
        self.fft = fft
        if fft:
            self.query_linear = nn.Linear(dim, dim)
            self.key_linear = nn.Linear(dim, dim)
            
    def forward(self, query, key):
        value = key
        if self.fft:
            query = torch.real(torch.fft.fft(query, dim=1))
            query = self.query_linear(query)
            key = self.key_linear(key)
            
        scaled_attention_scores = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)  # [B, 100, 4608]
        attention_probs = F.softmax(scaled_attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)  


        return output

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0., init_values=None,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp, fft=False,
                 seq_len=4608, num_prompt_tokens=200):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_norm=False, attn_drop =attn_drop, proj_drop = proj_drop, norm_layer = norm_layer)
        self.ls1 = LayerScale(dim, init_values=None) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.seq_len = seq_len

        self.down = nn.Linear(dim, 48)
        self._init_weights(self.down)
        self.gelu = nn.GELU()
        self.cross_attn = CrossAttention(dim=48, fft=fft)
        self.up = nn.Linear(48, dim)
        self.gate = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.gate, mean=1, std=0.02)

        self._init_weights(self.up)
        if fft:
            self._init_weights(self.cross_attn.query_linear)
            self._init_weights(self.cross_attn.key_linear)
        
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(dim, hidden_features = int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=None) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fft = fft
        self.num_prompt_tokens = num_prompt_tokens
        
        # Disable gradients for selected parameters to "freeze" parts of the model
        self._freeze_layers()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _freeze_layers(self):
        for layer in [self.norm1, self.attn, self.ls1, self.drop_path1, 
                      self.norm2, self.mlp, self.ls2, self.drop_path2]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        residual = x
        x = self.drop_path1(self.ls1(self.attn(self.norm1(x)))) + residual
        # print('!!!!!!!!!!!!!!!!!!', self.seq_len, self.num_prompt_tokens)
        x_fpt = self.gelu(self.down(x))
        x_fpt_prompt = x_fpt[:, -self.num_prompt_tokens:]
        x_fpt_modal = x_fpt[:, :self.seq_len]
        x_fpt_prompt = self.cross_attn(x_fpt_prompt, x_fpt_modal)
        x_fpt = torch.cat((x_fpt[:, :-self.num_prompt_tokens], x_fpt_prompt), dim=1)
        x_fpt = self.up(x_fpt) * self.gate

        x = self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) + x + x_fpt
        return x

class FPT(VisionTransformer):
    def __init__(self, model_name='B', modals=['rgb'], image_size=[768, 768], drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs):
        super().__init__(**kwargs)
        embed_dims = 768
        self.patch_size = 16
        self.num_modals = len(modals)
        self.rgb_adapter = PatchedInputAdapter(num_channels=3, patch_size_full=self.patch_size, stride_level=1)
        self.depth_adapter = PatchedInputAdapter(num_channels=3, patch_size_full=self.patch_size, stride_level=1)
        self.rgb_adapter.init(dim_tokens=embed_dims)
        if self.num_modals>1:
            self.depth_adapter.init(dim_tokens=embed_dims)

        seq_len = (image_size[0] // self.patch_size) * (image_size[1] // self.patch_size) * self.num_modals
        self.num_global_tokens = 1
        self.global_tokens = nn.Parameter(torch.zeros(1, self.num_global_tokens, embed_dims))
        trunc_normal_(self.global_tokens, std=0.02)
        
        self.proj_dec = nn.Linear(self.num_modals * embed_dims, 6144)
        
        self.num_prompt_tokens = 200
        self.prompt_dropout = Dropout(0.0)
        self.prompt_proj = nn.Identity()
        
        depth = 12
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dims, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None, proj_drop=0., attn_drop=0.,
                  drop_path=dpr[i], norm_layer=norm_layer, act_layer=nn.GELU, mlp_layer=Mlp, fft=i < 4, seq_len=seq_len, num_prompt_tokens=self.num_prompt_tokens)
            for i in range(depth)])
        

        
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_prompt_tokens, embed_dims))
        # trunc_normal_(self.deep_prompt_embeddings, std=math.sqrt(2. / (embed_dims + self.num_prompt_tokens)))
        val = math.sqrt(6. / float(3 * reduce(mul, _pair(self.patch_size), 1) + embed_dims))  # noqa

        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_prompt_tokens, embed_dims))

        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        
        for adapter in [self.rgb_adapter, self.depth_adapter]:
            for param in adapter.parameters():
                param.requires_grad = False
        self.global_tokens.requires_grad = False

    def adapt_tokens(self, encoder_tokens):
        split_tokens = torch.split(encoder_tokens, encoder_tokens.shape[1] // self.num_modals, dim=1)
        return torch.cat(split_tokens, dim=-1)

    def forward(self, inputs):
        B = inputs[0].shape[0]
        input_tokens = [self.rgb_adapter(inputs[0])]
        if self.num_modals > 1:
            input_tokens.append(self.depth_adapter(inputs[1]))
        input_tokens = torch.cat(input_tokens, dim=1)
        
        deep_prompt = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings.expand(B, -1, -1)))
        global_tokens = self.global_tokens.expand(B, -1, -1)
        x = torch.cat([input_tokens, global_tokens, deep_prompt], dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        x = x[:, :-self.num_global_tokens - self.num_prompt_tokens]
        x = self.adapt_tokens(x)
        return self.proj_dec(x)
    
if __name__ == '__main__':
    modals = ['img', 'depth']
    model = FPT('B', modals)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params/1e6)