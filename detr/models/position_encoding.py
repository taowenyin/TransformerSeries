# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from detr.util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        :param num_pos_feats: 位置的特征数，默认情况是128，X是128维，Y是128为，
                              即每个位置使用128维的数据表示，所以合计256维
        :param temperature:
        :param normalize:
        :param scale:
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # 这个NestedTensor简单说就是把{tensor, mask}打包在一起，tensor就是我们的图片的值，那么mask是对应的掩码
    # 假设有两张图片分别是[3, 200, 200]和[3, 200, 250]，当一个batch中的图片大小不一样的时候，我们要把它们处
    # 理的整齐，简单说就是把图片都padding成最大的尺寸，padding的方式就是补零，那么batch中的每一张图就有一个mask
    # 矩阵，所以mask大小为[2, 200,250], 在img有值的地方是1，其他地方是0，tensor大小为[2,3,200,250]是经过
    # padding后的结果
    def forward(self, tensor_list: NestedTensor):
        # 获取输入X和对应的Mask，Mask默认情况下，有值的地方为True，其他地方为False
        # Shape [B, C, H, W]
        x = tensor_list.tensors
        # Shape [B, H, W]
        mask = tensor_list.mask
        assert mask is not None
        # 对Mask取反，即有值的地方为False，其他地方为True
        not_mask = ~mask
        # 给Y轴和X轴设置位置
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # 因为最后一个数是前面的和，因此使用该值做归一化
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Shape [1, 128]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # x_embed [B, H, W] -> x_embed[:, :, :, None] [B, H, W, 1] -> pos_x [B, H, W, num_pos_feats]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # Shape [B, H, W, num_pos_feats]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # Shape [B, H, W, num_pos_feats + num_pos_feats] -> [B, num_pos_feats + num_pos_feats, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        # 这里使用了nn.Embedding，这是一个矩阵类，里面初始化了一个随机矩阵，矩阵的
        # 长是字典的大小，宽是用来表示字典中每个元素的属性向量。
        # 向量的维度根据你想要表示的元素的复杂度而定。类实例化之后可以根据字典中元素的
        # 下标来查找元素对应的向量。输入下标0，输出就是embeds矩阵中第0行。
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        # 对X、Y的位置进行编码，变成可学习的对象
        # x_emb：(w, 128)
        # y_emb：(h, 128)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        # pos：(h,w,256) → (256,h,w) → (1,256,h,w) → (b,256,h,w)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
