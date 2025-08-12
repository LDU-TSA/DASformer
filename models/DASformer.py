__all__ = ['DASformer.py']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.DASformer_backbone import DASformer_backbone
from layers.DASformer_layers import series_decomp
class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,patch_lengths: Optional[int] = None,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 head_type='flatten', verbose: bool = False, **kwargs):
        super().__init__()

        # 加载参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        patch_lengths = configs.patch_lengths
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        kernel_size = configs.kernel_size
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        self.cycle_len = configs.cycle
        cycle_len = configs.cycle  # 例如，周期长度为24
        channel_size = configs.enc_in

        # 模型组件
        self.linear = nn.Linear(seq_len, pred_len)
        self.decomp_module = series_decomp(cycle_len=cycle_len, channel_size=channel_size)
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # 引入一维卷积
        self.conv1d = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=3, padding=1)

        self.model_trend = DASformer_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                              strides=stride,patch_lens = patch_lengths,
                                             max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                             n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                             attn_dropout=attn_dropout,
                                             dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                             padding_var=padding_var,
                                             attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                             store_attn=store_attn,
                                             pe=pe, learn_pe=learn_pe, head_dropout=0, padding_patch=padding_patch,
                                             head_type=head_type, individual=individual, revin=revin, affine=affine,
                                             subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x, cycle_index):  # x: [Batch, Input length, Channel]

        trend_init, res_init = self.decomp_module(x, cycle_index)
        trend_init = trend_init.permute(0, 2, 1)
        trend = self.model_trend(trend_init)
        season = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        trend = trend.permute(0, 2, 1)
        y = trend + season
        return y
