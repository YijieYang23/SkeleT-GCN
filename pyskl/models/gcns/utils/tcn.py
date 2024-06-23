import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn import build_activation_layer
from mmcv.utils import _BatchNorm
from .init_func import bn_init,conv_init




class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)


class mstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass

class MSTCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 act_cfg=dict(type='ReLU'),
                 tcn_dropout=0):

        super().__init__()
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        branch_channels_rem = out_channels - branch_channels * (self.num_branches - 1)

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                build_activation_layer(act_cfg),
                unit_tcn(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            build_activation_layer(act_cfg),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels_rem, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels_rem)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(tcn_dropout)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        out = self.act(out)
        out = self.drop(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, _BatchNorm):
                bn_init(m, 1)

class dgmstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 num_joints=25,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = nn.ReLU()
        self.num_joints = num_joints
        # the size of add_coeff can be smaller than the actual num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])
        feat = local_feat + global_feat

        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)
