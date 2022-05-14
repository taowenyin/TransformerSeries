import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # 1、Q和K的转置相乘得到Attention，并除以Scale（temperature）
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # 2、如果有Mask那么就是Masked Multi-Head Attention，
        # 如果没有Mask，那么就是Multi-Head Attention。如果有
        # Mask，则把Attention与Mask进行Mask操作
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # 3、经过Softmax得到真正的Attention
        attn = self.dropout(F.softmax(attn, dim=-1))
        # 4、再把Attention和V相乘得到O
        output = torch.matmul(attn, v)

        return output, attn
