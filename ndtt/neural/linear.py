import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

class LinearZero(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Largely reuse torch.Linear but handle 0 in/out dimension
    Examples::
        >>> m = nn.Linear(4, 5, [0], [2])
        >>> input = torch.rand(128, 4) 
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 5])
        >>> print(output) # 2-th dim is 0
    """
    __constants__ = ['in_dim', 'out_dim', 'in_zero', 'out_zero']

    def __init__(self, name, in_dim, out_dim, in_zero, out_zero, device=None):
        super(LinearZero, self).__init__()
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        """
        pytorch (esp. autograd) doesn't properly handle 0-dim tensor
        so we augment in_dim and out_dim to > 0 
        for any dimension that is augmented 
        we keep the parameters to be 0
        s.t. it is equivalent to not using that dimension
        in_zero : list of indices that are augmented
        out_zero : list of indices that are augmented
        """
        self.in_zero = in_zero
        self.out_zero = out_zero
        assert in_dim >= 1, "in dim must > 0"
        assert out_dim >= 1, "out dim must > 0"

        self.weight = Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = Parameter(torch.Tensor(out_dim))

        self.weight_mask = torch.ones(out_dim, in_dim, device=device)
        self.bias_mask = torch.ones(out_dim, device=device)
        for i in self.in_zero: 
            self.weight_mask[:, i] = 0.0
        for o in self.out_zero: 
            self.weight_mask[o, :] = 0.0
            self.bias_mask[o] = 0.0
        
        self.reset_parameters()
        device = device or 'cpu'
        self.device = torch.device(device)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight * self.weight_mask, self.bias * self.bias_mask)
    
    def extra_repr(self):
        return '{} : in_dim={}, out_dim={}, in_zero={}, out_zero={}'.format(
            self.name, self.in_dim, self.out_dim, self.in_zero, self.out_zero
        )

    def count_params(self): 
        """
        not all params are actually used (some are augmented to facilitate pytorch)
        only count params that are used
        """
        num_bias = self.out_dim - len(self.out_zero)
        num_weight = (self.in_dim - len(self.in_zero)) * num_bias
        return num_weight + num_bias

    def get_upperbound(self): 
        """
        return a value that can't be exceeded no matter what input you have
        assume input entries are in [-1, +1]
        """
        tmp_weight = torch.abs(self.weight) * self.weight_mask
        tmp_bias = self.bias * self.bias_mask
        rst = tmp_weight.sum(-1) + tmp_bias
        return rst.detach()

class LinearCell(LinearZero): 
    """Linear transformation in continuous-time LSTM Cell: Mei and Eisner 2017
    its output dimension is 6 * cell_dim, because there are 6 diff gates
    """
    __constants__ =  ['in_dim', 'cell_dim', 'in_zero', 'cell_zero']

    def __init__(self, name, in_dim, cell_dim, in_zero, cell_zero, device=None):
        self.cell_dim = cell_dim
        self.cell_zero = cell_zero
        out_dim = cell_dim * 6
        out_zero = []
        for i in range(6): 
            out_zero += [ c + i * cell_dim for c in cell_zero ]
        super(LinearCell, self).__init__(name, in_dim, out_dim, in_zero, out_zero, device)
        assert cell_dim >= 1, "cell dim must > 0"

    def extra_repr(self): 
        return '{} : in_dim={}, out_dim={}, cell_dim={}, in_zero={}, out_zero={}, cell_zero={}'.format(
            self.name, self.in_dim, self.out_dim, self.cell_dim, self.in_zero, self.out_zero, self.cell_zero
        )
        