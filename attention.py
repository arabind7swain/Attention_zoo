import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor, BoolTensor
from typing import Optional


class Attention(nn.Module):
    """
    Single-head scaled dot-product self-attention
    """
    def __init__(self, d_model: int = 512, head_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = head_dim
        self.dim_K = head_dim
        self.query = nn.Linear(d_model, head_dim, bias=True)
        self.key  = nn.Linear(d_model, head_dim, bias=True)
        self.value = nn.Linear(d_model, head_dim, bias=True)

    def self_attention(self, Q: Tensor, K: Tensor, V: Tensor,
                       attn_mask: Optional[BoolTensor]=None) -> Tensor:
        """
        Perform self-attention on the input tensors.

        This is a simple implementation of self-attention that uses the dot product attention mechanism.

        Args:
            Q, K, V: (Batch_size, seq_Length, D_model)
            attn_mask: (B, L, L) or broadcastable
                       True  -> keep
                       False -> mask
        Returns:
            The output tensor of the self-attention layer.
        """
        K_T = torch.transpose(K, -1, -2) # [Batch, Seq, Dim] -> [Batch, Dim, Seq]
        score = torch.matmul(Q, K_T)     # Matmul: [B, L, D] x [B, D, L] -> [B, L, L]
        score /= math.sqrt(self.dim_K)   # Scale
        if attn_mask is not None:        # Mask (opt.)
            score = score.masked_fill(~attn_mask, -torch.inf)
        score = torch.softmax(score, dim=-1)        # SoftMax
        Z = torch.matmul(score, V)       # Matmul: [B, L, L] x [B, L, D] -> [B, L, D]
        return Z

    def forward(self, x: Tensor, attn_mask: Optional[BoolTensor]=None) -> Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Z = self.self_attention(Q, K, V, attn_mask)
        # Z = F.scaled_dot_product_attention(Q, K, V)
        return Z


class MultiheadAttention(nn.Module):
    r"""
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, d_model: int = 512, head_dim: int = 64, n_head:int=8) -> None:
        super().__init__()
        assert d_model == head_dim * n_head

        self.n_head = n_head
        self.head_dim = head_dim
        self.dim_K = head_dim

        self.proj = nn.Linear(head_dim * n_head, d_model, bias=False)

        self.multihead = nn.ModuleList([
            Attention(d_model, head_dim) for _ in range(n_head)
        ])

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[BoolTensor] = None,
        causal: bool = False,
    ) -> Tensor:
        """
        x: (Batch_size, seq_Length, D_model)
        padding_mask: (B, L)
            True  -> valid token
            False -> PAD
        causal: whether to apply causal (look-ahead) mask
        """
        B, L, _ = x.shape # Batch_size, seq_Length
        device = x.device

        attn_mask = None

        # Padding mask
        if padding_mask is not None:
            # (B, L) -> (B, 1, L) -> broadcast to (B, L, L)
            attn_mask = padding_mask[:, None, :]

        # Causal mask
        if causal:
            causal_mask = torch.tril(
                torch.ones(L, L, device=device)
            ).bool()  # (L, L)

            attn_mask = (
                causal_mask
                if attn_mask is None
                else causal_mask & attn_mask
            )

        # Apply all heads
        Z_s = torch.cat(
            [head(x, attn_mask) for head in self.multihead],
            dim=2, # concat on feature dim
        )  # (B, L, d_model)
        Z = self.proj(Z_s)
        return Z


class  MultiQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/1911.02150.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_query:int=8) -> None:
        super().__init__(word_size, embed_dim)
        self.n_query = n_query
        self.proj = nn.Linear(in_features=embed_dim * n_query,
                              out_features=embed_dim, bias=False)
        delattr(self, 'query')
        self.querys = nn.ModuleList([
            nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
            for _ in range(n_query)
        ])
        self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def forward(self, x: Tensor, mask:Optional[BoolTensor]=None) -> Tensor:
        K = self.key(x)
        V = self.value(x)
        Z_s = torch.cat([
            self.self_attention(query(x), K, V, mask) for query in self.querys
        ], dim=2)
        Z = self.proj(Z_s)
        return Z


class  GroupedQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/2305.13245.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64,
                 n_grouped: int = 4, n_query_each_group:int=2) -> None:
        super().__init__(word_size, embed_dim)
        delattr(self, 'query')
        delattr(self, 'key')
        delattr(self, 'value')

        self.grouped = nn.ModuleList([
            MultiQueryAttention(word_size, embed_dim, n_query=n_query_each_group)
            for _ in range(n_grouped)
        ])
        self.proj = nn.Linear(in_features=embed_dim * n_grouped,
                              out_features=embed_dim, bias=False)

    def forward(self, x: Tensor, mask:Optional[BoolTensor]=None) -> Tensor:
        Z_s = torch.cat([head(x, mask) for head in self.grouped], dim=2)
        Z = self.proj(Z_s)
        return Z
    
class CausalSelfAttentionIHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        
        # Standard Linear Projections
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Inter-Head Mixing Matrices (Learnable parameters)
        # Initialized to zeros; the network will learn how to route info between heads
        self.q_mix = nn.Parameter(torch.zeros(self.n_head, self.n_head))
        self.k_mix = nn.Parameter(torch.zeros(self.n_head, self.n_head))
        self.v_mix = nn.Parameter(torch.zeros(self.n_head, self.n_head))

    def _fuse_mix(self, weight, mix):
        """
        Dynamically folds the cross-head mixing matrix into the linear projection weights.
        weight shape: (n_embd, n_embd) -> reshaped to (n_head, head_dim * n_embd)
        mix shape: (n_head, n_head)
        """
        d = self.head_dim
        H = self.n_head
        # Matrix multiplication applies the cross-head mixing to the projection weights
        return (mix @ weight.view(H, d, -1).flatten(1)).view_as(weight)

    def forward(self, x):
        B, T, C = x.size()
        
        # 1. Fuse mixing matrices with projection weights on the fly
        q_weight = self._fuse_mix(self.c_q.weight, self.q_mix)
        k_weight = self._fuse_mix(self.c_k.weight, self.k_mix)
        v_weight = self._fuse_mix(self.c_v.weight, self.v_mix)
        
        # 2. Project inputs using the fused weights
        # Reshape to (B, T, H, D) then transpose to (B, H, T, D) for attention
        q = F.linear(x, q_weight).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = F.linear(x, k_weight).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = F.linear(x, v_weight).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 3. Standard Causal Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # 4. Reassemble and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

