from typing import Optional, Tuple

import numpy as np

import torch
from torch import Tensor

import torch.nn as nn
from torch.nn import Parameter, ReLU
import torch.nn.functional as F

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor

from torch_scatter import scatter

from torch_sparse import SparseTensor, matmul

class MANSDCL(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, args, in_channels: int, out_channels: int, 
                 snpf_md: int, snpf_ld: int, snpf_ml: int, device,
                 bias: bool = False, cached: bool = False, 
                 add_self_loops: bool = True, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.k = args.k
        self.mf_md1 = Linear(args.n_dis, args.rank, bias=False, weight_initializer='glorot')
        self.mf_md2 = Linear(args.n_miRNA, args.rank, bias=False, weight_initializer='glorot')
        self.mf_ld1 = Linear(args.n_dis, args.rank, bias=False, weight_initializer='glorot')
        self.mf_ld2 = Linear(args.n_lncRNA, args.rank, bias=False, weight_initializer='glorot')
        self.mf_lm1 = Linear(args.n_miRNA, args.rank, bias=False, weight_initializer='glorot')
        self.mf_lm2 = Linear(args.n_lncRNA, args.rank, bias=False, weight_initializer='glorot')
        
        log_sigmas = self.weight_init(snpf_md, snpf_ld, snpf_ml)
        self.log_sigma_md = Parameter(torch.tensor(log_sigmas[0], dtype=torch.float), requires_grad=True)
        self.log_sigma_ld = Parameter(torch.tensor(log_sigmas[1], dtype=torch.float), requires_grad=True)
        self.log_sigma_lm = Parameter(torch.tensor(log_sigmas[2], dtype=torch.float), requires_grad=True)

        # miRNA
        self.beta_miRNA = Parameter(torch.tensor([0.5], dtype=torch.float), requires_grad=True)
        self.gamma_miRNA = Parameter(torch.tensor([1], dtype=torch.float), requires_grad=True)
        self.lin1_miRNA = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.lin2_miRNA = Linear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        self.cl_miRNA = CL(in_dim=out_channels, out_dim=out_channels, alpha=0.5)

        # lncRNA
        self.beta_lncRNA = Parameter(torch.tensor([0.5], dtype=torch.float), requires_grad=True)
        self.gamma_lncRNA = Parameter(torch.tensor([1], dtype=torch.float), requires_grad=True)
        self.lin1_lncRNA = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.lin2_lncRNA = Linear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        self.cl_lncRNA = CL(in_dim=out_channels, out_dim=out_channels, alpha=0.5)

        # disease
        self.beta_dis = Parameter(torch.tensor([0.5], dtype=torch.float), requires_grad=True)
        self.gamma_dis = Parameter(torch.tensor([1], dtype=torch.float), requires_grad=True)
        self.lin1_dis = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.lin2_dis = Linear(out_channels, out_channels, bias=False, weight_initializer='glorot')
        self.cl_dis = CL(in_dim=out_channels, out_dim=out_channels, alpha=0.5)

        # others
        self.dropout = args.dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index_miRNA = None
        self._cached_edge_index_lncRNA = None
        self._cached_edge_index_dis = None
        self._cached_adj_t = None
        self.relu = ReLU()
        self.reg_params = (
            list(self.lin1_miRNA.parameters()) +
            list(self.lin2_miRNA.parameters()) +
            list(self.lin1_lncRNA.parameters()) +
            list(self.lin2_lncRNA.parameters()) +
            list(self.lin1_dis.parameters()) +
            list(self.lin2_dis.parameters())
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.device = device
        self.to(self.device)
        self.reset_parameters()

    def weight_init(self, snpf_md, snpf_ld, snpf_ml):
        weights_init = np.array([snpf_md, snpf_ld, snpf_ml], dtype=np.float32) / (snpf_md + snpf_ld + snpf_ml)
        weights_raw = np.exp(weights_init)
        weights_norm = weights_raw / weights_raw.sum()
        log_sigmas = -0.5 * np.log(weights_norm)
        return log_sigmas

    def reset_parameters(self):
        self.lin1_miRNA.reset_parameters()
        self.lin2_miRNA.reset_parameters()
        self.lin1_lncRNA.reset_parameters()
        self.lin2_lncRNA.reset_parameters()
        self.lin1_dis.reset_parameters()
        self.lin2_dis.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index_miRNA = None
        self._cached_edge_index_lncRNA = None
        self._cached_edge_index_dis = None
        self._cached_adj_t = None

    def forward(self, md: Tensor, ld: Tensor, lm: Tensor, 
                edge_index_miRNA: Adj, edge_index_lncRNA: Adj, edge_index_dis: Adj) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        miRNA1 = self.mf_md1(md)
        dis1 = self.mf_md2(md.t())
        lncRNA1 = self.mf_ld1(ld)
        dis2 = self.mf_ld2(ld.t())
        lncRNA2 = self.mf_lm1(lm)
        miRNA2 = self.mf_lm2(lm.t())

        miRNA = miRNA1 +  miRNA2
        lncRNA = lncRNA1 + lncRNA2
        dis = dis1 + dis2

        miRNA_repr, miRNA_cl = self.onestage_forward(
            miRNA, edge_index_miRNA, self._cached_edge_index_miRNA,
            self.beta_miRNA, self.gamma_miRNA, self.lin1_miRNA, self.lin2_miRNA, self.cl_miRNA
        )

        lncRNA_repr, lncRNA_cl = self.onestage_forward(
            lncRNA, edge_index_lncRNA, self._cached_edge_index_lncRNA,
            self.beta_lncRNA, self.gamma_lncRNA, self.lin1_lncRNA, self.lin2_lncRNA, self.cl_lncRNA
        )
        dis_repr, dis_cl = self.onestage_forward(
            dis, edge_index_dis, self._cached_edge_index_dis,
            self.beta_dis, self.gamma_dis, self.lin1_dis, self.lin2_dis, self.cl_dis
        )

        out_md = torch.matmul(miRNA_repr, dis_repr.t())
        out_ld = torch.matmul(lncRNA_repr, dis_repr.t())
        out_ml = torch.matmul(lncRNA_repr, miRNA_repr.t())

        out_md = (out_md - out_md.mean()) / (out_md.std() + 1e-6)
        out_ld = (out_ld - out_ld.mean()) / (out_ld.std() + 1e-6)
        out_ml = (out_ml - out_ml.mean()) / (out_ml.std() + 1e-6)

        return out_md, out_ld, out_ml, miRNA_cl, lncRNA_cl, dis_cl


    def onestage_forward(self, x: Tensor, edge_index: Adj, cached_edge_index: Optional[Tuple[Tensor, Tensor]],
                        beta: Parameter, gamma: Parameter, lin1: Linear, lin2: Linear, cl,
                        edge_weight: OptTensor = None) -> Tuple[Tensor, Tensor]:
        if cached_edge_index is None:
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), 
                                             False, self.add_self_loops, dtype=x.dtype)
            edge_index2, edge_weight2 = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), 
                                               False, False, dtype=x.dtype)
            if self.cached:
                cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = cached_edge_index[0], cached_edge_index[1]
        ew2 = edge_weight2.view(-1, 1)

        x = lin1(x)
        x = self.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = lin2(x)
        x = self.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        h = x.clone()

        g = self.cal_g_gradient(edge_index2, x, edge_weight=ew2)
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, [x.size(0), x.size(0)])
        Ax = torch.spmm(adj, x)
        Gx = torch.spmm(adj, g)
        x =  (1 - beta) * x + beta * Ax  + beta * gamma * Gx
        cl_loss = cl(x, h)       
        return x, cl_loss

    def cal_g_gradient(self, edge_index, x, edge_weight=None):
        row, col = edge_index[0], edge_index[1]
        onestep = scatter((x[col] - x[row]) * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='mean')
        onestep = self.feature_norm(onestep)
        twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='mean')
        twostep = self.feature_norm(twostep)
        return twostep
    
    def feature_norm(self, fea):
        device = fea.device
        epsilon = 1e-12
        fea_sum = torch.norm(fea, p=1, dim=1)
        fea_inv = 1 / np.maximum(fea_sum.detach().cpu().numpy(), epsilon)
        fea_inv = torch.from_numpy(fea_inv).to(device)
        fea_norm = fea * fea_inv.view(-1, 1)
        return fea_norm
    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'


class CL(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, alpha: float = 0.5):
        super(CL, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.tau = 0.5 
        self.fc1 = nn.Linear(in_dim, in_dim // 2)
        self.fc2 = nn.Linear(in_dim // 2, out_dim)

    def projection(self, z: Tensor) -> Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        sim_matrix = torch.mm(z1, z2.t()) / self.tau
        labels = torch.arange(sim_matrix.size(0), device=z1.device)

        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.t(), labels)
        loss = (loss_1 + loss_2) / 2.0
        return loss
    
    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        return self.sim(h1, h2)