''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        :param n_head: Head的数量
        :param d_model: 输入数据的维度
        :param d_k: Key的维度
        :param d_v: Value的维度
        :param dropout:
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 把Q映射到n_head x Q的维度
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # 把K映射到n_head x K的维度
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # 把V映射到n_head x V的维度
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # 最后把n_head x d_v重新映射到原始维度
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # K、V的维度，Head的大小
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # Batch_Size，Q、K、V的长度
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # 要加上原先的Q进行残差
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # 把输入的Q、K、V映射后，进行维度调整为（Batch， Length， Head数量， QKV的维度）
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 转化维度，变为（Batch， Head数量, Length， QKV的维度）
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 如果有Mask，则为Mask-MultiHead-Attention，否则就是MultiHead-Attention
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # 通过ScaledDotProductAttention得到Query（Output）和Attention
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # 把Q从维度（Batch， Head数量, Length， QKV的维度）变回（Batch， Length， Head数量， QKV的维度），
        # 然后把Head数量和QKV的维度进行合并，得到（Batch， Length， Head数量 x QKV的维度）
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # 与残差连接
        q += residual

        # 再做层归一化
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        :param d_in: 经过MultiHeadAttention后的维度作为输入的维度
        :param d_hid: 中间层的维度
        :param dropout:
        """
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
