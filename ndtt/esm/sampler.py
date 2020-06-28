# -*- coding: utf-8 -*-
# !/usr/bin/python
import numpy 

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
__author__ = 'Hongyuan Mei'


class DownSampler(object):
    """
    down sample a set of types from the large event type space 
    """
    def __init__(self, mode, down_sample_size, device=None):
        super(DownSampler, self).__init__()
        self.mode = mode 
        self.down_sample_size = down_sample_size
        if self.mode == 'none': 
            self.down_sample_size = -1
        device = device or 'cpu'
        self.device = torch.device(device)

    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def cpu(self):
        self.device = torch.device('cpu')
        super().cuda(self.device)


    def sample(self, support): 
        if self.mode == 'none': 
            # no down sampling 
            return support, 1.0

        if len(support) <= 0: 
            # no support at all
            return support, 1.0
        
        total_size = len(support)
        ratio = 1.0 * total_size / self.down_sample_size

        if self.mode == 'uniform': 
            probs = torch.ones(total_size, dtype=torch.float32, device=self.device)
            indices = torch.multinomial(probs, self.down_sample_size, replacement=True)
        else: 
            raise Exception(f"Unknown down sampling mode : {self.mode}")
        
        sampled_event_types = [
            support[int(i)] for i in indices
        ]
        
        return sampled_event_types, ratio
