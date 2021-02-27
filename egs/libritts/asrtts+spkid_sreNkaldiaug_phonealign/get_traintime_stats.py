#!/usr/bin/env python3

import sys
import numpy as np

def isfloat(element):
    try:
        float(element)
        return True
    except:
        return False

file_in = sys.argv[1]

period_list = []
for line in open(file_in):
    temp = line.strip().split()
    ele_list = []
    for ele in temp:
        if isfloat(ele): # assuming the first ele is a number
            ele_list.append(float(ele))
        else: # this is a boundary between two different epochs
            for i in range(len(ele_list)-1):
                period_list.append(ele_list[i+1] - ele_list[i])
            ele_list = []

if len(ele_list) > 1:
    for i in range(len(ele_list)-1):
        period_list.append(ele_list[i+1] - ele_list[i])

print("min: {}".format(np.min(period_list)))
print("max: {}".format(np.max(period_list)))
print("mean: {}".format(np.mean(period_list)))
print("std: {}".format(np.std(period_list)))

