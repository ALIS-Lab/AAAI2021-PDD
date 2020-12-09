import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat
import numpy as np
from collections import Counter


def count_top_K_grid(arr, K=3):
    c = Counter(arr)
    c_sort = sorted(c.items(), key=lambda x:x[1])
    top_K_grid = [c_sort[-i][0] for i in range(1, K+1)]
    return top_K_grid


def replace_arr_by_count(arr, M=10):
    c = Counter(arr)
    aux_c = np.zeros(M)
    aux_c[list(c.keys())] = list(c.values())
    arr_r = aux_c[arr]
    return arr_r


class Parallel_Dropout(InplaceFunction):
    
    @staticmethod
    def _make_noise(con_fc):
        return con_fc.new().resize_as_(con_fc)

    @classmethod
    def forward(cls, ctx, con_fc, K=3, M=10, alpha=0.9, beta=0.1, train=False, inplace=False):
        ctx.train = train
        ctx.inplace = inplace
        
        if not ctx.train:
            return con_fc

        if ctx.inplace:
            ctx.mark_dirty(con_fc)
            output = con_fc
        else:   
            output = con_fc.clone()

        con_fc = con_fc.detach().cpu().numpy()
        
        max_con = np.max(con_fc, axis=1, keepdims=True)
        min_con = np.min(con_fc, axis=1, keepdims=True)

        con_feat_dif = np.floor((con_fc - min_con) * M / (max_con - min_con)).astype(int)
        con_feat_dif[con_feat_dif==M] = M -1
        top_K_inx = np.apply_along_axis(count_top_K_grid, 1, con_feat_dif)
   
        feat_dif_list = np.split(con_feat_dif, K, axis=1)

        fc_sample_pb = []
        
        for i in range(K):
     
            fc_i_feat_dif = feat_dif_list[i]
            fc_i_p_mask = fc_i_feat_dif == top_K_inx[:,i].reshape(-1,1)
            fc_i_non_mask = fc_i_feat_dif != top_K_inx[:,i].reshape(-1,1)

            fc_i_r = np.apply_along_axis(replace_arr_by_count, 1, fc_i_feat_dif)
            fc_i_p = beta * (1 - (fc_i_r / fc_i_r.shape[1])) * fc_i_non_mask
            fc_i_p[fc_i_p_mask] = alpha

            fc_i_sample_pb = torch.FloatTensor(np.random.binomial(1, fc_i_p, fc_i_p.shape))
            fc_sample_pb.append(fc_i_sample_pb)

        ctx.noise = torch.cat(fc_sample_pb, dim=1).cuda()
        output.mul_(ctx.noise)

        return output


    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * ctx.noise, None, None, None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None, None, None







    
