# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
organize logs
@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime
from itertools import chain

from ndtt.io.log import LogBatchReader

import argparse
__author__ = 'Hongyuan Mei'


def main():

    parser = argparse.ArgumentParser(description='organize logs')


    parser.add_argument(
        '-d', '--Domain', required=True, type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )

    args = parser.parse_args()

    path_storage = os.path.abspath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)

    org = LogBatchReader(args.PathDomain)
    org.writeCSV()


if __name__ == "__main__": main()
