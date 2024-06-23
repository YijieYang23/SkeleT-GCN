import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from .init_func import bn_init, conv_init

EPS = 1e-4


class grouped_mapping_framework(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 M,
                 M_adaptive,
                 same_transformation_importance=True,
                 gcn_type='unit_gcn',
                 split_way='conv+split',
                 concat_way='concat+conv',
                 norm='BN',
                 act='ReLU',
                 **gcn_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = A.size(0)
        self.inner_inc = in_channels // self.num_groups
        self.inner_outc = out_channels // self.num_groups
        basic_gcn_blocks = []
        assert gcn_type in ['unit_gcn', 'dggcn', 'unit_ctrgcn']
        assert split_way in ['conv+split', 'split', 'conv+repeat']
        assert concat_way in ['concat+conv', 'concat']
        self.split_way = split_way
        self.concat_way = concat_way

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.split_way == 'conv+split':
            if self.num_groups != 1:
                self.conv_pre = nn.Sequential(
                    nn.Conv2d(in_channels, self.inner_inc * self.num_groups, 1),
                    build_norm_layer(self.norm_cfg, self.inner_inc * self.num_groups)[1],
                    self.act)

        elif self.split_way == 'conv+repeat':
            if self.num_groups != 1:
                self.conv_pre = nn.Sequential(
                    nn.Conv2d(in_channels, self.inner_inc, 1),
                    build_norm_layer(self.norm_cfg, self.inner_inc)[1],
                    self.act)

        self.conv_post = nn.Conv2d(self.inner_outc * self.num_groups, out_channels, 1)

        for i in range(self.num_groups):
            if gcn_type == 'unit_gcn':
                basic_gcn_blocks.append(unit_gcn(self.inner_inc, self.inner_outc, A[i], with_res=False, **gcn_kwargs))
            elif gcn_type == 'dggcn':
                basic_gcn_blocks.append(dggcn(self.inner_inc, self.inner_outc, A[i], with_res=False, **gcn_kwargs))
            elif gcn_type == 'unit_ctrgcn':
                basic_gcn_blocks.append(unit_ctrgcn(self.inner_inc, self.inner_outc, A[i], with_res=False, **gcn_kwargs))
            else:
                raise ValueError

        self.basic_gcn_blocks = nn.ModuleList(basic_gcn_blocks)
        assert M_adaptive in [None, 'init', 'offset', 'importance']
        self.M_adaptive = M_adaptive

        self.down = residual_block(in_channels, out_channels, M, M_adaptive, t_sample=None)

        if M is not None:
            if M.shape[0] == M.shape[1] and same_transformation_importance:
                self.M_adaptive = 'importance'
            M = M[None, :, :].repeat(self.num_groups, 1, 1)
            if self.M_adaptive == 'init':
                self.M = nn.Parameter(M.clone())
            else:
                self.register_buffer('M', M)

            if self.M_adaptive in ['offset', 'importance']:
                self.WM = nn.Parameter(M.clone())
                if self.M_adaptive == 'offset':
                    nn.init.uniform_(self.WM, -1e-6, 1e-6)
                elif self.M_adaptive == 'importance':
                    nn.init.constant_(self.WM, 1)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x)

        M = None
        if hasattr(self, 'M'):
            M_switch = {None: self.M, 'init': self.M}
            if hasattr(self, 'WM'):
                M_switch.update({'offset': self.M + self.WM, 'importance': self.M * self.WM})
            M = M_switch[self.M_adaptive]

        y = []

        if hasattr(self, 'conv_pre'):
            x = self.conv_pre(x)

        if self.split_way == 'conv+repeat':
            # x:(n c t v) tm:(k,v,w) or None
            for i in range(len(self.basic_gcn_blocks)):
                y.append(self.basic_gcn_blocks[i](torch.matmul(x, M[i]) if M is not None else x))
        else:
            x = x.view(n, self.num_groups, -1, t, v)

            if M is not None:
                x = torch.einsum('nkctu,kuv->nkctv', (x, M)).contiguous()

            for i in range(len(self.basic_gcn_blocks)):
                y.append(self.basic_gcn_blocks[i](x[:, i]))

        y = torch.cat(y, dim=1)  # k (n c t v) -> (n kc t v)
        y = self.conv_post(y)
        return self.act(self.bn(y) + res)

    def init_weights(self):
        pass


class residual_block(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 M=None,
                 M_adaptive=None,
                 t_sample=None,
                 t_factor=1,
                 kernel_size=1):
        super(residual_block, self).__init__()
        assert t_sample in [None, 'up_sample', 'down_sample']
        t_factor = 1 if t_sample is None else t_factor
        self.M_adaptive = M_adaptive
        if M is not None and M.shape[0] != M.shape[1]:
            if self.M_adaptive == 'init':
                self.M = nn.Parameter(M.clone())
            else:
                self.register_buffer('M', M)

            if self.M_adaptive in ['offset', 'importance']:
                self.WM = nn.Parameter(M.clone())
                if self.M_adaptive == 'offset':
                    nn.init.uniform_(self.WM, -1e-6, 1e-6)
                elif self.M_adaptive == 'importance':
                    nn.init.constant_(self.WM, 1)

        if in_channels != out_channels or t_sample is not None:
            pad = (kernel_size - 1) // 2
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(t_factor, 1), padding=(pad, 0)),
                build_norm_layer(dict(type='BN'), out_channels)[1])
        else:
            self.down = lambda x: x

    def forward(self, x):
        # n, c, t, v = x.shape
        if hasattr(self, 'M'):
            M_switch = {None: self.M, 'init': self.M}
            if hasattr(self, 'WM'):
                M_switch.update({'offset': self.M + self.WM, 'importance': self.M * self.WM})
            M = M_switch[self.M_adaptive]
            x = torch.einsum('nctu,uv->nctv', (x, M)).contiguous()
        return self.down(x)


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='importance',
                 conv_pos='pre',
                 with_res=True,
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0
        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)

    def init_weights(self):
        pass


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


class unit_ctrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, with_res=True, **kwargs):

        super(unit_ctrgcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None

        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)


class dggcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=None,
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU',
                 with_res=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        self.subset_wise = subset_wise

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # Introduce non-linear
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        res = self.down(x) if hasattr(self, 'down') else 0
        A = self.A

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)

            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
            ada_graph = getattr(self, self.ctr_act)(diff)

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
            ada_graph = getattr(self, self.ada_act)(ada_graph)

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A = ada_graph + A

        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()

        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        return self.act(self.bn(x) + res)
