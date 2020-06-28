# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
draw learning curve 
@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import fnmatch
import csv
import datetime
from itertools import chain

from ndtt.eval.draw import Drawer

import argparse
__author__ = 'Hongyuan Mei'

def main():

    parser = argparse.ArgumentParser(description='draw learning curve')

    parser.add_argument(
        '-d', '--Domain', required=True, type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-fn', '--FileName', required=True, type=str,
        help='file name of csv that has the results?'
    )
    parser.add_argument(
        '-g', '--Gold', action='store_true', 
        help='show groundtruth?'
    )
    parser.add_argument(
        '-b', '--Band', action='store_true', 
        help='draw band?'
    )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )

    args = parser.parse_args()

    path_storage = os.path.abspath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)

    path_csv = os.path.join(args.PathDomain, f'{args.FileName}.csv')

    """
    read csv 
    """
    def _read_csv(path_csv): 
        with open(path_csv, 'r', encoding='utf-8-sig') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader, None) 
            columns = {}
            for h in header: 
                columns[h] = []
            for row in csvreader: 
                for h, v in zip(header, row): 
                    if h != 'seqs': 
                        v = float(v)
                    columns[h].append(v)
        return columns

    columns = _read_csv(path_csv)

    dr = Drawer()

    path_save = os.path.join(args.PathDomain, 'figures')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    dr.drawLearningCurve(
        columns=columns, 
        figname=f"{args.Domain}_{args.FileName}", 
        draw_gold=args.Gold,
        draw_band=args.Band, 
        path_save=path_save )


if __name__ == "__main__": main()