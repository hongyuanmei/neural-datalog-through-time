import torch
from torch import nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import time

class MPManager(object):

    def __init__(self, num_workers):
        """
        manage a single-instruction-multiple-data (SIMD) scheme
        :param int num_workers: The number of processors to run.
        """
        mp.set_start_method('spawn')
        # Counting the current batch size
        self.num_workers = num_workers
        # A pool of processes
        self.pool = mp.Pool(processes=num_workers)

    def run(self, function, arguments):
        """
        :param function : the instruction
        :param arguments : list of things processors loop over
        can be anything the function works on, e.g. model + data
        """
        output_and_grads = self.pool.map(function, arguments)
        return output_and_grads


