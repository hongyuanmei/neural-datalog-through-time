import torch
import torch.nn as nn
import torch.nn.functional as F

class Tanh(nn.Tanh): 
    """Modified nn.Tanh
    add count_params method
    """
    def __init__(self):
        super(Tanh, self).__init__()
    
    def count_params(self): 
        """
        gate has no params
        """
        return 0

class Softplus(nn.Softplus): 
    """Modified nn.Softplus
    add count_params method
    """
    def __init__(self):
        super(Softplus, self).__init__() # use default specs
    
    def count_params(self): 
        """
        gate has no params
        """
        return 0