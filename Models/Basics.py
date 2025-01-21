import torch
import torch.nn as nn
import torch.nn.functional as func
from einops import rearrange
from jaxtyping import Float, Bool
from typing import List, Literal, Tuple
from inspect import getfullargspec
import math

Tensor = torch.Tensor


class Swish(nn.Module):
    def forward(self, x):
        return x * func.sigmoid(x)


class FeatureNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        # x is of shape (B, L, D)
        mean = x.mean(dim=1, keepdim=True)  # Mean along the token dimension (L)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # Variance along L
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Apply learnable scale (gamma) and shift (beta)
        return self.gamma * x_norm + self.beta


class AttentionBlock(nn.Module):
    def __init__(self, d_in: int, d_head: int, d_expand: int, d_out: int, n_heads: int,
                 dropout: float=0.0, score: Literal["prod", "dist"] = "dist", d_time: int = 0):
        super(AttentionBlock, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.d_expand = d_expand
        self.score = score
        self.d_time = d_time
        self.scale = nn.Parameter(torch.tensor(d_head if score == "prod" else 1.414, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

        self.d_q = d_head * n_heads
        self.d_k = d_head * n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q + self.d_k + self.d_v

        self.qkv_proj = nn.Sequential(
            PositionalEncoding(d_in),
            FeatureNorm(d_in),
            Swish(),
            nn.Linear(d_in, d_qkv),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        if d_time == 0:
            self.ff = nn.Sequential(
                nn.LayerNorm(d_in), Swish(),
                nn.Linear(d_in, d_expand),
                Swish(), nn.Dropout(dropout),
                nn.Linear(d_expand, d_out),
            )
            torch.nn.init.zeros_(self.ff[-1].weight)
            torch.nn.init.zeros_(self.ff[-1].bias)
        else:
            self.time_proj = nn.Sequential(
                nn.Linear(d_time, d_time),
                Swish(),
                nn.Linear(d_time, d_expand)
            )
            self.ff1 = nn.Sequential(nn.LayerNorm(d_in), Swish(), nn.Linear(d_in, d_expand))
            self.ff2 = nn.Sequential(Swish(), nn.Dropout(dropout), nn.Linear(d_expand, d_out))
            torch.nn.init.zeros_(self.ff2[-1].weight)
            torch.nn.init.zeros_(self.ff2[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, t=None):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(x).split([self.d_q, self.d_k, self.d_v], dim=-1)
        q = rearrange(q, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, head_c)
        k = rearrange(k, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, in_c)

        if self.score == "prod":
            attn = torch.softmax(q @ k.transpose(-1, -2) * (self.scale ** -0.5), dim=-1)
        else:
            attn = torch.softmax(- torch.cdist(q, k) ** 2 / self.scale ** 2, dim=-1)
            # attn = torch.softmax(torch.exp(- torch.cdist(q, k) ** 2 / self.scale ** 2), dim=-1)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        if self.d_time == 0:
            return self.ff(x) + self.shortcut(x)
        else:
            return self.ff2(self.ff1(x) + self.time_proj(t)) + self.shortcut(x)


class CrossAttnBlock(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_context: int,
                 d_head: int,
                 d_expand: int,
                 d_out: int,
                 n_heads: int,
                 dropout: float=0.1,
                 score: Literal["prod", "dist"] = "dist",
                 d_time: int = 0):
        super(CrossAttnBlock, self).__init__()

        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.score = score
        self.d_time = d_time
        self.scale = nn.Parameter(torch.tensor(d_head if score == "prod" else 1.414, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Sequential(
            PositionalEncoding(d_in),
            nn.LayerNorm(d_in), Swish(),
            nn.Linear(d_in, d_head * n_heads),
        )

        self.kv_proj = nn.Sequential(
            PositionalEncoding(d_context, 512),
            nn.LayerNorm(d_context), Swish(),
            nn.Linear(d_context, (d_head + d_in) * n_heads),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        if d_time == 0:
            self.ff = nn.Sequential(
                nn.LayerNorm(d_in), Swish(),
                nn.Linear(d_in, d_expand),
                Swish(), nn.Dropout(dropout),
                nn.Linear(d_expand, d_out),
            )
            torch.nn.init.zeros_(self.ff[-1].weight)
            torch.nn.init.zeros_(self.ff[-1].bias)
        else:
            self.time_proj = nn.Sequential(
                nn.Linear(d_time, d_time),
                Swish(),
                nn.Linear(d_time, d_in + d_expand)
            )
            self.ff1 = nn.Sequential(nn.LayerNorm(d_in), Swish(), nn.Linear(d_in, d_expand))
            self.ff2 = nn.Sequential(Swish(), nn.Dropout(dropout), nn.Linear(d_expand, d_out))
            torch.nn.init.zeros_(self.ff2[-1].weight)
            torch.nn.init.zeros_(self.ff2[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, context, t=None):
        B, N, in_c = x.shape

        q = rearrange(self.q_proj(x), 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)

        k, v = self.kv_proj(context).split([self.d_head * self.H, in_c * self.H], dim=-1)
        k = rearrange(k, 'B N (H C) -> (B H) N C', H=self.H)   # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        if self.score == "prod":
            attn = torch.softmax(q @ k.transpose(-1, -2) * (self.scale  ** -0.5), dim=-1)
        else:
            attn = torch.softmax(torch.exp(- torch.cdist(q, k) ** 2 / self.scale ** 2), dim=-1)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        if self.d_time == 0:
            return self.ff(x) + self.shortcut(x)
        else:
            t_shift, t_scale = self.time_proj(t).split([self.d_in, self.d_expand], dim=-1)
            return self.ff2(self.ff1(x + t_shift) * torch.sigmoid(t_scale)) + self.shortcut(x)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Res2D(nn.Module):
    def __init__(self, d_in: int, d_mid: int, d_out: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(d_in, d_mid, 3, 1, 1),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv2d(d_mid, d_mid, 3, 1, 1),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv2d(d_mid, d_out, 3, 1, 1)
        )

        torch.nn.init.zeros_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv2d(d_in, d_out, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)



class SERes1D(nn.Module):
    def __init__(self, d_in: int, d_mid: int, d_out: int):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_in, d_in // 4, 1, 1, 0),
            Swish(),
            nn.Conv1d(d_in // 4, d_mid, 1, 1, 0),
            nn.Sigmoid()
        )

        self.s1 = nn.Sequential(
            nn.Conv1d(d_in, d_mid, 1, 1, 0),
            nn.InstanceNorm1d(d_mid),
            # nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv1d(d_mid, d_mid, 3, 1, 1))

        self.s2 = nn.Sequential(
            nn.InstanceNorm1d(d_mid),
            # nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv1d(d_mid, d_out, 1, 1, 0)
        )

        torch.nn.init.zeros_(self.s2[-1].weight)
        torch.nn.init.zeros_(self.s2[-1].bias)

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv1d(d_in, d_out, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.s2(self.s1(x) * self.se(x))



class Res1D(nn.Module):
    def __init__(self, d_in: int, d_mid: int, d_out: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(d_in, d_mid, 1, 1, 0),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv1d(d_mid, d_mid, 3, 1, 1),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv1d(d_mid, d_out, 1, 1, 0)
        )

        torch.nn.init.zeros_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv1d(d_in, d_out, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)

class Conv2dNormAct(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv2d(d_in, d_out, k, s, p),
            nn.GroupNorm(8, d_out),
            Swish()
        )


class Conv1dNormAct(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv1d(d_in, d_out, k, s, p),
            nn.GroupNorm(8, d_out),
            Swish()
        )


class SequentialWithAdditionalInputs(nn.Sequential):
    def forward(self, x, *args):
        for module in self:
            x = module(x, *args)
        return x


class Rearrange(nn.Module):
    def __init__(self, from_shape: str, to_shape: str, **kwargs):
        super().__init__()
        self.from_shape = from_shape
        self.to_shape = to_shape
        self.kwargs = kwargs

    def forward(self, x):
        return rearrange(x, self.from_shape + ' -> ' + self.to_shape, **self.kwargs)


def cacheArgumentsUponInit(original_init):
    # set an attribute with the same name and value as whatever is passed to __init__
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # get names of args
        argspec = getfullargspec(original_init)
        for i, arg in enumerate(argspec.args):
            if i < len(args):
                setattr(self, arg, args[i])

        original_init(self, *args, **kwargs)

    return __init__


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, d_enc: int = 32, max_len: int = 1024):
        """
        Implements positional encoding for sequence data.
        :param d_model: Dimensionality of the model/embedding.
        :param max_len: Maximum length of the sequence.
        """
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_enc)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_enc, 2).float() * (-math.log(10000.0) / d_enc))  # Shape: (d_model // 2)

        # Compute positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Register as a buffer to avoid updates during training
        self.pe = nn.Parameter(pe.unsqueeze(0))
        self.proj = nn.Linear(d_model + d_enc, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.
        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Tensor with positional encodings added, same shape as input.
        """
        B, L, D = x.shape
        return self.proj(torch.cat([x, self.pe[:, :L].expand(B, -1, -1)], dim=-1))