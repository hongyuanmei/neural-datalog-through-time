import torch
import torch.nn as nn
import torch.nn.functional as F

class Cell(object):
    """Cell block in (Neural) Datalog
    obtained by querying the database 
    only know they are active, don't know when they are created
    """
    def __init__(self, name, dimension, zeros, see_edges, is_event):
        self.name = name
        self.dimension = dimension
        self.zeros = zeros
        self.see_edges = see_edges
        self.is_event = is_event

    def __repr__(self):
        s0 = f'Cell : '
        s1 = f'name={self.name}, dimension={self.dimension}, zeros={self.zeros}, is event={self.is_event}'
        s2 = f', see edges={self.see_edges}'
        return s0+s1+s2