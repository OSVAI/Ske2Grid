import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoEncIndex(nn.Module):
    def __init__(self, num_jts, grid_shape, add_prior=None, temperature=30):
        super(AutoEncIndex, self).__init__()
        self.grid_shape = grid_shape
        self.J = num_jts
        self.temperature = temperature
        self.HW = np.prod(grid_shape)
        self.add_prior = add_prior
		
        self.register_parameter('index_sel', torch.nn.Parameter(torch.zeros(self.HW), requires_grad=False))
        self.register_parameter('sgt_trans_mat', torch.nn.Parameter(torch.rand(self.HW, self.J), requires_grad=True))

    def update_temperature(self, temperature):
        self.temperature = temperature

    def forward(self, use_gumbel_noise, is_training=False):
        sgt_trans_mat = self.sgt_trans_mat
        index_sel = self.index_sel

        if is_training:
            if use_gumbel_noise:
                #print(self.temperature)
                sgt_trans_mat_gumbel = F.gumbel_softmax(sgt_trans_mat, tau=self.temperature, hard=False, dim=-1)
                J_indicator = torch.ones(self.J)
                index = sgt_trans_mat_gumbel.topk(self.J, dim=-1)[1]
                for row_idx in range(self.HW):
                    if row_idx < self.J:
                        for col_idx in range(min(row_idx, self.J)+1):
                            if J_indicator[index[row_idx, col_idx]]==1:
                                index_sel[row_idx] = index[row_idx, col_idx]
                                J_indicator[index[row_idx, col_idx]] = 0
                                break
                    else:
                        index_sel[row_idx] = index[row_idx,0]
                y_hard = torch.zeros_like(sgt_trans_mat, memory_format=torch.legacy_contiguous_format).scatter_(-1, index_sel.unsqueeze(-1).long(),1.0)
                sgt_trans_mat_hard = y_hard - sgt_trans_mat.detach() + sgt_trans_mat
            else:
                J_indicator = torch.ones(self.J)
                index = sgt_trans_mat.topk(self.J, dim=-1)[1]
                for row_idx in range(self.HW):
                    if row_idx < self.J:
                        for col_idx in range(min(row_idx, self.J)+1):
                            if J_indicator[index[row_idx, col_idx]]==1:
                                index_sel[row_idx] = index[row_idx, col_idx]
                                J_indicator[index[row_idx, col_idx]] = 0
                                break
                    else:
                        index_sel[row_idx] = index[row_idx,0]
                y_hard = torch.zeros_like(sgt_trans_mat, memory_format=torch.legacy_contiguous_format).scatter_(-1, index_sel.unsqueeze(-1).long(),1.0)
                sgt_trans_mat_hard = y_hard - sgt_trans_mat.detach() + sgt_trans_mat
                #print(sgt_trans_mat_hard)
        else:
            J_indicator = torch.ones(self.J)
            index = sgt_trans_mat.topk(self.J, dim=-1)[1]
            for row_idx in range(self.HW):
                if row_idx < self.J:
                    for col_idx in range(min(row_idx, self.J)+1):
                        if J_indicator[index[row_idx, col_idx]]==1:
                            index_sel[row_idx] = index[row_idx, col_idx]
                            J_indicator[index[row_idx, col_idx]] = 0
                            break
                else:
                    index_sel[row_idx] = index[row_idx, 0]
            #print(index_sel)
            sgt_trans_mat_hard = F.one_hot(index_sel.long(), num_classes=self.J).float()

        return sgt_trans_mat_hard
