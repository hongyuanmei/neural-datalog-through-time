import torch
import torch.nn as nn
import torch.nn.functional as F

from ndtt.neural.linear import LinearZero as Linear


class CTLSTMGate(nn.Module): 
    """Modified gates of continuous-time LSTM cel: :Mei and Eisner 2017
    We get rid of the output gate as explained in our paper
    """
    def __init__(self, name, cell_dim, cell_zero, time_created, device=None):
        super(CTLSTMGate, self).__init__()
        device = device or 'cpu'
        self.device = torch.device(device)
        self.name = name
        self.cell_dim = cell_dim
        self.cell_zero = cell_zero
        assert cell_dim >= 1, "cell dim must > 0"
        self.cell_mask = torch.ones(cell_dim, device=device)
        for c in self.cell_zero: 
            self.cell_mask[c] = 0.0
        self.time_created = time_created

    def forward(self, input, c, c_b): 
        """
        input is the output of a LinearCell
        """
        assert input.dim() == 1, "only one instance to update"
        assert c.size(0) == c_b.size(0), "c dim should == c_b dim"
        in_dim = input.size(0)
        assert in_dim == 6 * self.cell_dim, f"in != 6 cell : in={in_dim}, cell={self.cell_dim}"
        
        g_i, g_f, z, g_i_b, g_f_b, g_d = input.chunk(6)
        g_i = torch.sigmoid(g_i)
        g_f = torch.sigmoid(g_f)
        z = torch.tanh(z)
        g_i_b = torch.sigmoid(g_i_b)
        g_f_b = torch.sigmoid(g_f_b)
        g_d = F.softplus(g_d, beta=1.0)

        c = g_f * c + g_i * z 
        c_b = g_f_b * c_b + g_i_b * z 

        c = c * self.cell_mask # mask out the augmented dimensions
        c_b = c_b * self.cell_mask
        g_d = g_d * self.cell_mask

        return {'start': c, 'target': c_b, 'decay': g_d}

    # def forward(self, input, c, c_b): 
    #     """
    #     input is the output of a LinearCell
    #     """
    #     assert input.dim() == 1, "only one instance to update"
    #     assert c.size(0) == c_b.size(0), "c dim should == c_b dim"
    #     in_dim = input.size(0)
    #     assert in_dim == 6 * self.cell_dim, f"in != 6 cell : in={in_dim}, cell={self.cell_dim}"
        
    #     ifzif = torch.sigmoid(input[:5*self.cell_dim])
    #     g_i, g_f, z, g_i_b, g_f_b = ifzif.chunk(5)
    #     z = 2 * z - 1.0
    #     g_d = F.softplus(input[5*self.cell_dim:], beta=1.0)

    #     c = g_f * c + g_i * z 
    #     c_b = g_f_b * c_b + g_i_b * z 

    #     c = c * self.cell_mask # mask out the augmented dimensions
    #     c_b = c_b * self.cell_mask
    #     g_d = g_d * self.cell_mask

    #     return {'start': c, 'target': c_b, 'decay': g_d}

    def extra_repr(self):
        rst = '{} : cell_dim={}, cell_zero={}, time_created={}'.format(
            self.name, self.cell_dim, self.cell_zero, self.time_created
        )
        return rst

    def count_params(self): 
        """
        gate has no params
        """
        return 0
