import torch
import torch.nn as nn
import torch.nn.functional as F


class Event(object): 

    def __init__(self, name, dimension, zeros, who_see_it, who_it_kill):
        """
        name : structured name of term
        who see it : terms that see this event when it happened
        """
        self.name = name
        self.dimension = dimension 
        self.zeros = zeros
        self.who_see_it = who_see_it
        self.who_see_it_set = set(who_see_it)
        self.who_it_kill = who_it_kill
        self.who_it_kill_set = set(who_it_kill)

    def __repr__(self):
        s0 = f'Event : '
        s1 = f'name={self.name}, dimension={self.dimension}, zeros={self.zeros}, who see it={self.who_see_it}'
        return s0+s1

