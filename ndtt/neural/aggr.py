import math
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class Aggregation(nn.Module):
    """Applies aggregation to a set of incoming vectors: :math:`y_i = s^{-1} \sum s(x_i)`
    s(x) = sign(x) |x|^{1/b}
    s^{-1}(y) = sign(y) |y|^{b}
    """

    def __init__(self, name, device=None):
        super(Aggregation, self).__init__()
        """
        it has only one parameter 
        in future, we may allow different b for different dimensions
        """
        self.name = name
        self.bias = Parameter(torch.Tensor(1))
        self.eps = numpy.finfo(float).eps
        self.max = torch.finfo().max
        self.reset_parameters()
        device = device or 'cpu'
        self.device = torch.device(device)

    def reset_parameters(self, x=0.0):
        init.constant_(self.bias, x)

    def use_sum(self, input): 
        return torch.sum(input, dim=0)

    def use_extreme(self, input): 
        ma, _ = torch.max(input, dim=0)
        mi, _ = torch.min(input, dim=0)
        ma_idx = (torch.abs(ma) >= torch.abs(mi))
        y = mi
        y[ma_idx] = ma[ma_idx]
        return y

    def use_beta(self, input): 
        """
        avoid overflow/underflow
        if input_max == 0.0, ratio is NaN : should avoid this
        """
        input_abs = torch.abs(input) + self.eps # avoid 0 in max
        input_max, max_idx = torch.max(input_abs, dim=0, keepdim=True)
        ratio = input_abs / input_max
        
        pooling = 1 + self.bias * self.bias
        ratio_pow = ratio ** pooling
        #print(f"ratio pow : {ratio_pow}")
        """
        elements indexed by max_idx must be 1 
        cuz they are 1 ** pooling : be robust to numerical computation
        """
        """
        # manually set 1 ** pooling to be 1 
        # uncomment it if we see NAN again
        # we don't set it since 0 ** 0 will be set to 1 anyway
        # we don't set it for speed up as well
        ratio_pow_flat_trans = torch.flatten(ratio_pow, start_dim=1).transpose(0, 1)
        #print(f"ratio pow flat : {ratio_pow_flat_trans}")
        max_idx_flat = torch.flatten(max_idx)
        s0 = ratio_pow_flat_trans.size(0)
        ratio_pow_flat_trans[torch.arange(s0), max_idx_flat] = 1.0
        #print(f"max idx flat : {max_idx_flat}")
        #print(f"ratio pow flat : {ratio_pow_flat_trans}")
        ratio_pow = ratio_pow_flat_trans.transpose(0,1).reshape(ratio_pow.size())
        #print(f"ratio pow : {ratio_pow}")
        """

        """
        sign can't be 0 : it over/under-flow otherwise
        y can't be 0 : cuz ratio has a 1 at least
        res can be 0 : cuz of input_max but not sign!
        """
        input_sign = torch.sign(input)
        """
        # manually set sign(0) to be 1 
        # uncomment it if we see NAN again
        # we don't set it so as to preserve 0 value
        # and speed up
        sign_zero = (torch.abs(input_sign) < 0.5)
        input_sign[sign_zero] = 1.0 
        #print(f"sign : {input_sign}")
        """

        y = torch.sum( ratio_pow * input_sign, dim=0)
        y_sign = torch.sign(y)
        """
        # manually set sign(0) to be 1 
        # uncomment it if we see NAN again
        # we don't set it so as to preserve 0 value
        # and speed up
        y_sign_zero = (torch.abs(y_sign) < 0.5)
        y_sign[y_sign_zero] = 1.0
        #print(f"y : {y}")
        #print(f"y sign : {y_sign}")
        #print(f"y abs : {torch.abs(y)}")
        """
        """
        y abs can't be 0 : y_abs ** 0 should go to 1, can't be 0
        """
        y_abs = torch.abs(y) + self.eps 
        res = input_max * ( y_abs ** (1/pooling) )
        res[res>self.max] = self.max
        res = y_sign * res
        return res.squeeze(0)

    def forward(self, input):
        """
        input is a tensor stacked (over dim=0) from a list 
        """
        return self.use_beta(input)

    def count_params(self): 
        """
        aggregator has only 1 param
        """
        return 1

    def extra_repr(self):
        return f'Aggregation : {self.name} : bias={self.bias}, pooling={1+self.bias*self.bias}'


class DecayWeight(nn.Module):
    """Applies aggregation to compute weights for decay gates
    """
    def __init__(self, name, device=None):
        super(DecayWeight, self).__init__()
        """
        it has only one parameter 
        in future, we may allow different b for different dimensions
        """
        self.name = name
        self.bias = Parameter(torch.Tensor(1))
        self.eps = numpy.finfo(float).eps
        self.max = torch.finfo().max
        self.reset_parameters()
        device = device or 'cpu'
        self.device = torch.device(device)

    def reset_parameters(self, x=0.0):
        init.constant_(self.bias, x)

    def use_beta(self, dco, dc, dcb): 
        abs_dco = torch.abs(dco) + self.eps
        abs_dc = torch.abs(dc) + self.eps 
        abs_dcb = torch.abs(dcb) + self.eps 

        max_abs_dc, _ = torch.max(abs_dc, dim=0, keepdim=True)
        max_abs_dcb, _ = torch.max(abs_dcb, dim=0, keepdim=True)

        pooling = 1 + self.bias * self.bias

        ratio_dc = (abs_dc / max_abs_dc) ** pooling
        ratio_dc = ratio_dc / torch.sum(ratio_dc, dim=0, keepdim=True)

        ratio_dcb = (abs_dcb / max_abs_dcb) ** pooling
        ratio_dcb = ratio_dcb / torch.sum(ratio_dcb, dim=0, keepdim=True)

        """
        naively sum up may overflow
        so we factor summand = large_factor * (summand / large_factor)
        we keep large factor for future use to normalize all weights
        """
        fac = torch.cat([abs_dco.unsqueeze(0), max_abs_dc, max_abs_dcb], dim=0)
        fac, _ = torch.max(fac, dim=0, keepdim=True)
        fac[fac>self.max] = self.max
        fac = fac.expand_as(abs_dc) # M x D

        abs_dco = abs_dco / fac 
        abs_dc = abs_dc / fac 
        abs_dcb = abs_dcb / fac

        sum_abs_dc = torch.sum(abs_dc, dim=0)
        sum_abs_dcb = torch.sum(abs_dcb, dim=0)

        w = abs_dco + sum_abs_dc * ratio_dc + sum_abs_dcb * ratio_dcb
        w[w > self.max] = self.max

        return fac, w

    def forward(self, dco, dc, dcb):
        """
        compute weight given dc and dcb 
        input is a tensor stacked (over dim=0) from a list 
        """
        """
        dco : dcell_old = c(t) - c(inf)
        dc : dcell = c - c(t)
        dcb : dcell_target = cb - c(inf)
        """
        return self.use_beta(dco, dc, dcb)

    def count_params(self): 
        """
        aggregator has only 1 param
        """
        return 1

    def extra_repr(self):
        return f'DecayWeight : {self.name} : bias={self.bias}, pooling={1+self.bias*self.bias}'


class AggregationDecay(nn.Module):
    """Applies aggregation to decay gate
    d = weighted harmonic mean of d_rm with w_rm
    """

    def __init__(self, device=None):
        super(AggregationDecay, self).__init__()
        """
        it has only one parameter 
        in future, we may allow different b for different dimensions
        """
        self.eps = numpy.finfo(float).eps
        device = device or 'cpu'
        self.device = torch.device(device)

    def forward(self, fac, w, d):
        """
        element-wise weighted harmonic mean
        input is a tensor stacked (over dim=0) from a list 
        """
        d += self.eps 
        """
        to avoid overflow/underflow, scale by max
        """
        max_fac, _ = torch.max(fac, dim=0, keepdim=True)
        norm_fac = fac / (max_fac + self.eps)
        max_w, _ = torch.max(w, dim=0, keepdim=True)
        norm_w = w / (max_w + self.eps)
        norm_w = norm_w * norm_fac + self.eps
        total_w = torch.sum(norm_w, dim=0, keepdim=True)
        norm_w = norm_w / total_w
        ans = 1.0 / torch.sum( norm_w / d , dim=0 )
        return ans

    def count_params(self): 
        """
        aggregator has only 0 param
        """
        return 0

    def extra_repr(self):
        return f'Aggregation for decay'