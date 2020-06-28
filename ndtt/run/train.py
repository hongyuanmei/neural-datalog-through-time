# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Train neural Datalog through time (NDTT)
using maximum likelihood estimation (MLE)

@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ndtt.esm.trainer import Trainer

import argparse
__author__ = 'Hongyuan Mei'


def get_args(): 

    parser = argparse.ArgumentParser(description='training neural Datalog through time (NDTT) using MLE')

    parser.add_argument(
        '-d', '--Domain', required=True, type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-db', '--Database', required=True, type=str, help='which database to use?'
    )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )
    parser.add_argument(
        '-cp', '--CheckPoint', default=-1, type=int, 
        help='every # tokens (>1) in a seq to accumulate gradients (and cut compute graph), -1 meaning entire seq'
    )
    parser.add_argument(
        '-bs', '--BatchSize', default=1, type=int, 
        help='# checkpoints / seqs to update parameters'
    )
    parser.add_argument(
        '-tp', '--TrackPeriod', default=1000, type=int, help='# seqs to train for each dev'
    )
    parser.add_argument(
        '-m', '--Multiplier', default=1, type=float,
        help='constant of N=O(I), where N is # of sampled time points for integral'
    )
    parser.add_argument(
        '-dm', '--DevMultiplier', default=1, type=int,
        help='constant of N=O(I), where N is # of sampled time points for integral'
    )
    parser.add_argument(
        '-tr', '--TrainRatio', default=1.0, type=float, help='fraction of training data to use'
    )
    parser.add_argument(
        '-dr', '--DevRatio', default=1.0, type=float, help='fraction of dev data to use'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', default=20, type=int, help='max # training epochs'
    )
    parser.add_argument(
        '-lr', '--LearnRate', default=1e-3, type=float, help='learning rate'
    )
    parser.add_argument(
        '-np', '--NumProcess', default=1, type=int, help='# of processes used, default is 1'
    )
    parser.add_argument(
        '-nt', '--NumThread', default=1, type=int, help='OMP NUM THREADS'
    )
    parser.add_argument(
        '-tdsm', '--TrainDownSampleMode', default='none', type=str, choices=['none', 'uniform'], 
        help='for training, how do you want to down sample it? none? uniform?'
    )
    parser.add_argument(
        '-tdss', '--TrainDownSampleSize', default=1, type=int, 
        help='for training, down sample size, 1 <= dss <= K'
    )
    parser.add_argument(
        '-ddsm', '--DevDownSampleMode', default='none', type=str, choices=['none', 'uniform'], 
        help='for dev, how do you want to down sample it? none? uniform?'
    )
    parser.add_argument(
        '-ddss', '--DevDownSampleSize', default=1, type=int, 
        help='for dev, down sample size, 1 <= dss <= K'
    )
    parser.add_argument(
        '-lp', '--LSTMPool', default='full', type=str, choices=['full', 'simp'], 
        help='for LSTM pooling, full(default):full-verison-in-paper;simp:a-simplification'
    )
    parser.add_argument(
        '-um', '--UpdateMode', default='sync', type=str, choices=['sync', 'async'], 
        help='way of updating lstm after computed new cells'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true', help='use GPU?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )
    args = parser.parse_args()
    return args

def aug_args_with_log(args): 
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    args.Version = torch.__version__
    args.ID = id_process
    args.TIME = time_current

    path_storage = os.path.abspath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)

    dict_args = vars(args)
    folder_name = get_foldername(dict_args)

    path_logs = os.path.join(args.PathDomain, 'Logs', folder_name)
    os.makedirs(path_logs)
    args.PathLog = os.path.join(path_logs, 'log.txt')
    args.PathSave = os.path.join(path_logs, 'saved_model')

    args.NumProcess = 1
    #if args.MultiProcessing: 
    #    if args.NumProcess < 1:
    #        args.NumProcess = os.cpu_count()
    
    if args.NumThread < 1: 
        args.NumThread = 1
    
    print(f"mp num threads in torch : {torch.get_num_threads()}")
    if torch.get_num_threads() != args.NumThread: 
        print(f"not equal to NumThread arg ({args.NumThread})")
        torch.set_num_threads(args.NumThread)
        print(f"set to {args.NumThread}")
        assert torch.get_num_threads() == args.NumThread, "not set yet?!"

def get_foldername(dict_args): 
    args_used_in_name = [
        ['Database', 'db'],
        ['CheckPoint', 'cp'], 
        ['TrainRatio', 'tr'], 
        ['Multiplier', 'm'], 
        ['TrainDownSampleMode', 'tdsm'],
        ['TrainDownSampleSize', 'tdss'],
        ['DevDownSampleMode', 'ddsm'],
        ['DevDownSampleSize', 'ddss'],
        ['LSTMPool', 'lp'], 
        ['UpdateMode', 'um'], 
        ['LearnRate', 'lr'],
        ['Seed', 'seed'],
        ['ID', 'id']
    ]
    folder_name = list()
    for arg_name, rename in args_used_in_name:
        folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
    folder_name = '_'.join(folder_name)
    return folder_name

def main():

    args = get_args()
    aug_args_with_log(args)
    trainer = Trainer(args)
    trainer.run()

if __name__ == "__main__": main()
