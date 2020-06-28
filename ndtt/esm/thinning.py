# -*- coding: utf-8 -*-
# !/usr/bin/python
import numpy 

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
__author__ = 'Hongyuan Mei'


class EventSampler(nn.Module):
    """
    sample one event from a (structure-)NHP using chosen algorithm
    """
    def __init__(self, num_sample, num_exp, device=None):
        super(EventSampler, self).__init__()
        self.num_sample = num_sample
        self.num_exp = num_exp
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

    def draw_next_time(self, arguments, mode='ordinary'): 
        if mode=='ordinary': 
            rst, weights = self.draw_next_time_ordinary(arguments)
        elif mode=='fractional': 
            raise NotImplementedError
        else: 
            raise Exception(f"Unknow sampling mode : {mode}")
        return rst, weights

    def draw_next_time_ordinary(self, arguments, fastapprox=True): 
        time_last_event, boundary, next_event_name, \
        types, ratio, \
        datalog, cdb, active = arguments
        """
        ordinary thinning algorithm (with a little minor approximation)
        """
        """
        sample some time points to get intensities 
        s.t. we can estimate a conservative upper bound
        """
        over_sample_rate = 5.0
        times_for_bound = torch.empty(
            size=[10], dtype=torch.float32, device=self.device
        ).uniform_( time_last_event, boundary )
        datalog.create_cache(times_for_bound)
        intensities_for_bound = datalog.compute_intensities(types, cdb, active)
        #print(f"intensity for bound is {intensities_for_bound}")
        bounds = intensities_for_bound.sum(dim=-1).max() * over_sample_rate
        #bounds = datalog.compute_intensity_bounds( types, cdb, active )
        # 1 * len(types)
        if next_event_name in types: 
            # actual next event gets sampled out
            sample_rate = bounds * ratio # size : 0, scalar
        else: 
            # samples don't have actual next event
            # we still count actual next event in to give a better estimate
            intensity_actual_for_bound = datalog.compute_intensities(
                [next_event_name], cdb, active)
            bound_actual = intensity_actual_for_bound.sum(dim=-1).max() * over_sample_rate
            sample_rate = bounds * (ratio - 1.0/len(types)) + bound_actual
        datalog.clear_cache() # clear cache of embeddings
        """
        estimate # of samples needed to cover the interval
        """
        #S = int( (boundary - time_last_event) * sample_rate * 1.2 + 1 )
        """
        bound is roughly C * intensity (C>1), so accept rate is 1/C
        if we have N samples, then prob of no accept is only (1 - 1/C)^N
        meaning that if we want accept at least one sample with 99% chance
        we need > log0.01 / log(1 - 1/C) samples 
        if we make C reasonable small, e.g., 
        C = 2, # of samples : 6
        C = 5, # of samples : 20
        C = 10, # of samples : 44
        therefore, some reasonable num_exp is good enough, e.g, 100 or 500
        if we use 100 samples, for each C, prob(at one sample) 
        C = 2, 99.99%
        C = 5, 99.99%
        C = 10, 99.99%
        a benefit of using large S : making sure accumulated times reach boundary
        in that case, if none is accepted, use the farthest time as prediction
        """
        S = self.num_exp # num_exp is usually large enough for 2 * intensity bound
        """
        prepare result
        """
        rst = torch.empty(
            size=[self.num_sample], dtype=torch.float32, device=self.device
        ).fill_(boundary) 
        # for those didn't accept proposed times, use boundary or even farther
        weights = torch.ones(
            size=[self.num_sample], dtype=torch.float32, device=self.device)
        weights /= weights.sum()
        """
        sample times : dt ~ Exp(sample_rate)
        compute intensities at these times
        """
        if fastapprox: 
            """
            reuse the proposed times to save computation (of intensities)
            different draws only differ by which of them is accepted
            but proposed times are same
            """
            Exp_numbers = torch.empty(
                size=[1, S], dtype=torch.float32, device=self.device)
        else: 
            Exp_numbers = torch.empty(
                size=[self.num_sample, S], dtype=torch.float32, device=self.device)
        Exp_numbers.exponential_(1.0)
        sampled_times = Exp_numbers / sample_rate 
        #print(f"sampled times : {sampled_times}")
        sampled_times = sampled_times.cumsum(dim=-1) + time_last_event
        #print(f"sampled times : {sampled_times}")
        datalog.create_cache(sampled_times)
        intensities_at_sampled_times = datalog.compute_intensities(types, cdb, active)
        if next_event_name in types: 
            total_intensities = intensities_at_sampled_times.sum(dim=-1) * ratio
        else: 
            intensity_actual = datalog.compute_intensities([next_event_name], cdb, active)
            total_intensities = \
                intensities_at_sampled_times.sum(dim=-1) * (ratio - 1.0/len(types)) + \
                    intensity_actual.sum(dim=-1)
        # size : N * S or 1 * S
        datalog.clear_cache() # clear cache of embeddings
        if fastapprox: 
            """
            reuse proposed times and intensities at those times
            """
            sampled_times = sampled_times.expand(self.num_sample, S)
            total_intensities = total_intensities.expand(self.num_sample, S)
        """
        randomly accept proposed times
        """
        Unif_numbers = torch.empty(
            size=[self.num_sample, S], dtype=torch.float32, device=self.device)
        Unif_numbers.uniform_(0.0, 1.0)
        criterion = Unif_numbers * sample_rate / total_intensities
        #print(f"criterion is {criterion}")
        """
        for each parallel draw, find its min criterion
        if that < 1.0, the 1st (i.e. smallest) sampled time with cri < 1.0 is accepted 
        if none is accepted, use boundary/maxsampletime for that draw
        """
        min_cri_each_draw, _ = criterion.min(dim=1)
        who_has_accepted_times = min_cri_each_draw < 1.0
        #print(f"who has accepted times : {who_has_accepted_times}")
        """
        whoever accepts times, find their accepted times
        """
        #print(f"sampled times : {sampled_times}")
        sampled_times_accepted = sampled_times.clone()
        sampled_times_accepted[criterion>=1.0] = sampled_times.max() + 1.0
        #print(f"sampled times : {sampled_times_accepted}")
        accepted_times_each_draw, accepted_id_each_draw = sampled_times_accepted.min(dim=-1)
        #print(f"accepted times : {accepted_times_each_draw}")
        # size : N
        #print(f"rst = {rst}")
        rst[who_has_accepted_times] = \
            accepted_times_each_draw[who_has_accepted_times]
        #print(f"rst = {rst}")
        who_not_accept = ~who_has_accepted_times
        who_reach_further = sampled_times[:, -1] > boundary
        rst[who_not_accept&who_reach_further] = \
            sampled_times[:, -1][who_not_accept&who_reach_further]
        return rst, weights