import torch
import torch.nn as nn
import torch.nn.functional as F

class Term(object):
    """Structured Term in (Neural) Datalog
    """
    def __init__(self, name, dimension, zeros, transform_edges, is_event):
        """
        name : structured name of term
        dimension : dimension of its embedding
        is_event : boolean, if this term is a event
        has_cell : boolean, if this term has a LSTM cell
        """
        self.name = name
        self.dimension = dimension
        self.zeros = zeros
        self.transform_edges = transform_edges
        self.is_event = is_event

    def __repr__(self):
        s0 = f'Term : '
        s1 = f'name={self.name}, dimension={self.dimension}, zeros={self.zeros}, is event={self.is_event}'
        s2 = f', transform edges={self.transform_edges}'
        return s0+s1+s2