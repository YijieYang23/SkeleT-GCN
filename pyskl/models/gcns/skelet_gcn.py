import copy as cp
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import load_checkpoint
from ..builder import BACKBONES
from ...utils.graph import Graph
from .utils import mstcn, residual_block, grouped_mapping_framework, MSTCN, dgmstcn
from .utils import mapping_matrices
from einops import rearrange
import numpy as np

EPS = 1e-4


class InstancePoolingModule(nn.Module):

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_point=65,
                 drop_path=0.1,
                 pe=None,
                 **kwargs):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(in_channels, base_channels),
            nn.BatchNorm1d(base_channels),
            nn.ReLU())
        self.concat_pool_fc = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels),
            nn.BatchNorm1d(base_channels))

        self.group_pool_fc = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels),
            nn.BatchNorm1d(base_channels),
            nn.GELU())

        self.gelu = nn.GELU()

        assert pe in [None, 'V', 'VC']
        self.pe = pe
        if pe == 'V':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_point, 1))
        if pe == 'VC':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_point, base_channels))

        self.drop_path = drop_path

        self.poo1d = nn.AdaptiveMaxPool1d(1)
        self.poo3d = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        """Defines the computation performed at every call."""
        N, I, V, C, T = x.size()
        x = self.point_encoder(rearrange(x, 'n i v c t -> (n t v i) c'))  # (n t v i) c

        if self.pe is not None:
            x = rearrange(x, '(n t v i) c -> (n t i) v c', v=V, t=T, i=I)  # (n t v i) c
            x += self.pos_embedding
            x = rearrange(x, '(n t i) v c -> (n t v i) c', v=V, t=T, i=I)

        residual = x
        if self.training:
            drop_path = torch.rand(1) < self.drop_path
        else:
            drop_path = False
        if drop_path:
            residual = residual.new_zeros(residual.shape)

        x = rearrange(x, '(n t v i) c -> n c t v i', v=V, t=T, i=I)  # n c t v i
        # concat_pool
        x1 = x  # n c t v i

        x2 = self.poo1d(rearrange(x, 'n c t v i -> (n t v) c i'))  # (n t v) c 1
        x2 = rearrange(x2, '(n t v) c i -> n c t v i', v=V, t=T)  # n c t v 1
        x2 = x2.repeat(1, 1, 1, 1, I)  # n c t v i
        x = torch.cat([x1, x2], dim=1)  # n 2c t v i
        x = self.concat_pool_fc(rearrange(x, 'n c t v i -> (n t v i) c'))  # (n t v i) c
        x = self.gelu(x + residual)
        x = rearrange(x, '(n t v i) c -> n c t v i', v=V, t=T, i=I)  # n c t v i
        # group_pool
        x1 = self.poo3d(x).squeeze(-1)  # n c 1 1
        x1 = x1.repeat(1, 1, T, V)  # n c t v
        x2 = self.poo1d(rearrange(x, 'n c t v i -> (n t v) c i')).squeeze(-1)  # (n t v) c
        x2 = rearrange(x2, '(n t v) c -> n c t v', v=V, t=T)  # n c t v
        x = torch.cat([x1, x2], dim=1)  # n 2c t v
        x = self.group_pool_fc(rearrange(x, 'n c t v -> (n t v) c'))  # (n t v) c
        x = rearrange(x, '(n t v) c -> n c t v', v=V, t=T)  # n c t v
        return x


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = torch.sum(A, dim)
    h, w = A.shape
    Dn = torch.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = torch.mm(A, Dn)
    return AD


def random_graph(num_node, num_filter=8, init_std=.04, init_off=.02):
    return np.random.randn(num_filter, num_node, num_node) * init_std + init_off


class GraphTools(object):

    def __init__(self, init_graph_cfg, same_transformation=False, num_subsets=3):
        self.init_graph = Graph(**init_graph_cfg)
        self.same_transformation = same_transformation
        self.num_subsets = num_subsets
        self.M = mapping_matrices
        self.init_A = torch.tensor(self.init_graph.A, dtype=torch.float32, requires_grad=False)
        self.A_dict = dict()
        self.A_dict[f'J{self.init_graph.num_node}'] = normalize_digraph(self.init_A)

    def get_A(self, joints, num_groups):
        if f'J{joints}' not in self.A_dict:
            m = self.M[f'J{self.init_graph.num_node}toJ{joints}'].clone()
            target_A = torch.matmul(torch.matmul(torch.linalg.pinv(m), self.init_A), m)
            target_A = normalize_digraph(target_A)
            self.A_dict[f'J{joints}'] = torch.tensor(target_A, dtype=torch.float32, requires_grad=False)

        return self.A_dict[f'J{joints}'][None, None, :, :].repeat(num_groups, self.num_subsets, 1, 1).clone()

    def get_M(self, pre_joints, target_joints):
        if pre_joints == target_joints and not self.same_transformation:
            return None
        else:
            return self.M[f'J{pre_joints}toJ{target_joints}'].clone()


class STBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 M=None,
                 M_adaptive=None,
                 t_sample=None,
                 t_factor=1,
                 residual=True,
                 same_transformation_importance=False,  # * Only used when same_transformation == True
                 split_way='conv+split',
                 concat_way='concat+conv',
                 **kwargs):
        super().__init__()
        assert t_sample in [None, 'down_sample']
        t_factor = 1 if t_sample is None else t_factor
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'
        tcn_type = tcn_kwargs.pop('type', 'mstcn')
        assert tcn_type in ['mstcn', 'MSTCN', 'dgmstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn', 'dggcn', 'unit_ctrgcn']
        self.gcn = grouped_mapping_framework(in_channels, out_channels, A, M, M_adaptive,
                               same_transformation_importance, gcn_type,
                               split_way, concat_way, **gcn_kwargs)

        if tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=t_factor, **tcn_kwargs)
        elif tcn_type == 'dgmstcn':
            self.tcn = dgmstcn(out_channels, out_channels, stride=t_factor, num_joints=A.shape[-1], **tcn_kwargs)
        else:
            self.tcn = MSTCN(
                out_channels,
                out_channels,
                stride=t_factor,
                kernel_size=5,
                dilations=[1, 2],
                residual=False,
                tcn_dropout=0)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = residual_block(in_channels, out_channels, M, M_adaptive, t_sample, t_factor)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


@BACKBONES.register_module()
class SkeleT_GCN(nn.Module):

    def __init__(self,
                 init_graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pooling_stages=[5, 8],
                 pooling_joints=[27, 11],
                 num_groups=[1, 1, 1, 1, 2, 2, 2, 4, 4, 4],
                 same_transformation=False,
                 same_transformation_importance=False,  # * Only used when same_transformation == True
                 M_adaptive=None,
                 num_subsets=3,
                 split_way='split',
                 concat_way='concat+conv',
                 with_instance_pooling=False,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        assert len(pooling_stages) == len(pooling_joints)
        self.graph_tools = GraphTools(init_graph_cfg, same_transformation, num_subsets)

        self.data_bn_type = data_bn_type
        ip_kwargs = {k[3:]: v for k, v in kwargs.items() if k[:3] == 'ip_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:3] not in ['ip_']}
        self.kwargs = kwargs
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * self.graph_tools.init_graph.num_node)
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * self.graph_tools.init_graph.num_node)
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.stage_joints = [self.graph_tools.init_graph.num_node]
        for i in range(1, num_stages):
            if i + 1 in pooling_stages:
                idx = pooling_stages.index(i + 1)
                self.stage_joints.append(pooling_joints[idx])
            else:
                self.stage_joints.append(self.stage_joints[i - 1])

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.pooling_stages = pooling_stages
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        if with_instance_pooling:
            self.instance_pooling_block = InstancePoolingModule(in_channels, base_channels, self.stage_joints[0], **ip_kwargs)
        self.pool = nn.AdaptiveAvgPool2d(1)

        modules = []

        if self.in_channels != self.base_channels:
            modules = [STBlock(base_channels if with_instance_pooling else in_channels, base_channels,
                               self.graph_tools.get_A(self.stage_joints[0], num_groups[0]),
                               self.graph_tools.get_M(self.stage_joints[0], self.stage_joints[0]) if with_instance_pooling else None,
                               M_adaptive if with_instance_pooling else None,
                               t_sample=None, residual=True if with_instance_pooling else False, same_transformation_importance=same_transformation_importance,
                               split_way=split_way,
                               concat_way=concat_way,
                               **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels

            if stride == 1:
                modules.append(STBlock(in_channels, out_channels,
                                       self.graph_tools.get_A(self.stage_joints[i - 1], num_groups[i - 1]),
                                       self.graph_tools.get_M(self.stage_joints[i - 2], self.stage_joints[i - 1]), M_adaptive,
                                       t_sample=None, t_factor=stride, residual=True, same_transformation_importance=same_transformation_importance,
                                       split_way=split_way,
                                       concat_way=concat_way,
                                       **lw_kwargs[i - 1]))
            else:
                modules.append(STBlock(in_channels, out_channels,
                                       self.graph_tools.get_A(self.stage_joints[i - 1], num_groups[i - 1]),
                                       self.graph_tools.get_M(self.stage_joints[i - 2], self.stage_joints[i - 1]), M_adaptive,
                                       t_sample='down_sample', t_factor=stride, residual=True, same_transformation_importance=same_transformation_importance,
                                       split_way=split_way,
                                       concat_way=concat_way,
                                       **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained


    def init_weights(self):
        if isinstance(self.pretrained, str):
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, I, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, I * V * C, T))
        else:
            x = self.data_bn(x.view(N * I, V * C, T))
        if hasattr(self, 'instance_pooling_block'):
            x = x.view(N, I, V, C, T)
            x = self.instance_pooling_block(x)
        else:
            x = x.view(N, I, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * I, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)
        if hasattr(self, 'instance_pooling_block'):
            x = self.pool(x)  # n c 1 1
            x = x.reshape(N, -1)
        else:
            x = x.reshape((N, I) + x.shape[1:])
        return x
