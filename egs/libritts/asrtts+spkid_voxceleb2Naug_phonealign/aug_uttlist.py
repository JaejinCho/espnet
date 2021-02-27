#!/usr/bin/env python3

# This script generates augmented uttlist with "babble\|music\|noise\|reverb"
# Usage: aug_uttlist.py [ori_uttlist] [aug_uttlist]
# example: aug_uttlist.py ${f_uttlist} ${f_uttlist}_aug

import sys

f_uttlist_ori = sys.argv[1]
f_uttlist_aug = sys.argv[2]

noise_list = ['babble', 'music', 'noise', 'reverb']

uttlist = []
for line in open(f_uttlist_ori):
    uttlist.append(line.strip())

fopen = open(f_uttlist_aug,'w')
for utt in uttlist:
    fopen.write(utt+'\n')
    for noise in noise_list:
        fopen.write(utt+ '-' + noise + '\n')

fopen.close()

