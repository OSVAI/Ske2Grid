import torch
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint,_load_checkpoint
from mmcv.utils import _BatchNorm

from ...utils import get_root_logger, Graph
from ..builder import BACKBONES

import numpy as np
import torch.nn.functional as F

from .auto_grid_encode import AutoEncIndex


def zero(x):
    """return zero."""
    return 0


def identity(x):
    """return input itself."""
    return x


class STGCNBlock2CN(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out},
            V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn2cn = ConvTemporalGraphical2CN(in_channels, out_channels,
                                               kernel_size[1])
        self.tcn2cn = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), padding), nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = identity

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Defines the computation performed at every call."""
        n, c, t, h, w = x.size()
        x_temp = x.view(n, c, t, h * w)
        res = self.residual(x_temp)
        x = self.gcn2cn(x)  # nm x c x t x h x w
        nm, c, t, h, w = x.size()
        x = x.view(nm, c, t, h * w)
        x = self.tcn2cn(x)
        nm, c, t, hw = x.size()
        x = x.reshape(nm, c, t, h, w)
        if not isinstance(res, int):
            res = res.reshape(nm, c, t, h, w)
        x = x + res

        return self.relu(x)


class ConvTemporalGraphical2CN(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution.
            Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides
            of the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=1,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(t_padding, t_padding),
            padding_mode='zeros',  # zeros;circular;reflect;replicate
            stride=t_stride,  # (t_stride,1)
            dilation=(t_dilation, 1),
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Defines the computation performed at every call."""
        # assert x.size(0)
        nm, c, t, h, w = x.size()

        x = x.permute(0, 2, 1, 3, 4)  # nm,c,t,h,w --> nm,t,c,h,w
        x = x.reshape(nm * t, c, h, w)  # nm,t,c,h,w --> nmt,c,h,w

        x = self.conv(x)  # nmt,c_out,h,w
        x = self.bn(x)  # v0+bn

        nmt, cout, h, w = x.size()
        x = x.reshape(nm, t, cout, h, w)  # -->nm,t_c_out,h,w
        x = x.permute(0, 2, 1, 3, 4)  # -->nm,c_out,t,h,w

        return x.contiguous()


@BACKBONES.register_module()
class STGCN2CN_UpSmp_AutoEncode_Grid_Progressive(nn.Module):
    """Backbone of Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data.
        graph_cfg (dict): The arguments for building the graph.
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph. Default: True.
        data_bn (bool): If 'True', adds data normalization to the inputs.
            Default: True.
        pretrained (str | None): Name of pretrained model.
        **kwargs (optional): Other parameters for graph convolution units.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 graph_cfg,
                 grid_shape=(6,6),
                 progressive_joint_seq=[17,25],
                 data_bn=True,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else identity

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock2CN(
                in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            STGCNBlock2CN(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock2CN(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock2CN(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock2CN(64, 128, kernel_size, 2, **kwargs),
            STGCNBlock2CN(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock2CN(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock2CN(128, 256, kernel_size, 2, **kwargs),
            STGCNBlock2CN(256, 256, kernel_size, 1, **kwargs),
            STGCNBlock2CN(256, 256, kernel_size, 1, **kwargs),
        ))

        self.pretrained = pretrained
        self.grid_shape = grid_shape
        self.HW = np.prod(self.grid_shape)
        self.progressive_joint_seq = progressive_joint_seq #[17,25]
        self.J = A.size()[-1]
        self.graph_adjacency_mat = torch.zeros(self.J, self.J).float().cuda()
        for idx in range(len(self.graph.edge)):
            self.graph_adjacency_mat[self.graph.edge[idx]] = 1

        # Number of progressive stages.
        self.num_stage = len(self.progressive_joint_seq)
        # e.g. for g17->g25, self.num_jts_sampling_seq=[17, 25]
        self.num_jts_seq = self.progressive_joint_seq + [self.HW]

        # Stacked list for graph upsampling parameter
        self.sampling_stages = []
        for stage_i in range(self.num_stage):
            cur_num_jts_in = self.num_jts_seq[stage_i]
            cur_num_jts_out = self.num_jts_seq[stage_i+1]
            self.sampling_stages.append(
                nn.Parameter(torch.cat((torch.eye(cur_num_jts_in), torch.softmax(torch.rand(cur_num_jts_out - cur_num_jts_in, cur_num_jts_in), dim=-1)), 0))
            )
        self.sampling_stages = nn.ParameterList(self.sampling_stages)

        # Stacked list for SGT transformation matrix
        self.SGT_stages = []
        for stage_i in range(self.num_stage):
            cur_num_grid = self.num_jts_seq[stage_i+1]
            self.SGT_stages.append(AutoEncIndex(num_jts=cur_num_grid,grid_shape=(int(np.sqrt(cur_num_grid)),int(np.sqrt(cur_num_grid)))))
        self.SGT_stages = nn.ModuleList(self.SGT_stages)


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for stage_i in range(self.num_stage - 1):
            self.sampling_stages[stage_i].requires_grad = False
            self.SGT_stages[stage_i].sgt_trans_mat.requires_grad = False

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # data normalization
        x = x.float()
        # n, c, t, v, m = x.size()  # bs 3 300 25(17) 2
        n, m, t, v, c = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # N M V C T
        x = x.view(n * m, v * c, t)  # nm x vc x t
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)  # n m v c t
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)  # bsx2 3 300 25(17)

        x = x.permute(0, 3, 1, 2)
        x = x.view(n * m, v, c * t)  # 2b,17,900

        for stage_i in range(self.num_stage):
            cur_num_jts_in = self.num_jts_seq[stage_i]     # e.g. 17 for 1st progressive layer; 25 for 2nd progressive layer.
            # Graph upsampling phase
            cur_upsampling_matrix = self.sampling_stages[stage_i]  # e.g. 25*17
            # if cur_num_jts_in == 17:
            if stage_i == 0:
                cur_upsampling_matrix = torch.mm(cur_upsampling_matrix, self.graph_adjacency_mat)  # 25*17
            x = torch.bmm(cur_upsampling_matrix.repeat(n*m, 1, 1), x)

            # SGT permutating phase
            cur_SGT_matrix = self.SGT_stages[stage_i](use_gumbel_noise=False, is_training=True)
            x = torch.bmm(cur_SGT_matrix.repeat([n*m, 1, 1]), x)

        x = x.view(n * m, self.grid_shape[0] * self.grid_shape[1], c, t)  # 2b,25,3,300
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(n * m, c, t, self.grid_shape[0], self.grid_shape[1])

        # forward
        for gcn in self.st_gcn_networks:
            x = gcn(x)

        return x

