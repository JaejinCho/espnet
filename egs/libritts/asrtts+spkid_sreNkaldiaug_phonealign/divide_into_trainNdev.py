#!/usr/bin/env python3

# Description: This script filters an augmented file (e.g., one of files that is composed of "[uttid] [some value]" format in each line such as utt2spk, etc.) based on uttlist.
# Usage: thisscript.py [uttlist_dev_portion] [augmented file]
# e.g.) divide_into_trainNdev.py ./data/sre_tel/sre_tel_dev.uttlist ./data/sre_tel_aug_1m/utt2spk # output will be ./data/sre_tel_aug_1m/utt2spk.{train,dev}
import sys

fpath_uttlist_dev= sys.argv[1]
fpath_augfile = sys.argv[2] # e.g., some utt2spk file

# get uttlist for dev portion
uttlist_dev = []
for line in open(fpath_uttlist_dev):
    uttlist_dev.append(line.strip())

# write to files ()
fout_train_handle = open(fpath_augfile + '.train','w')
fout_dev_handle = open(fpath_augfile + '.dev','w')

for line in open(fpath_augfile):
    uttid = line.strip().split(' ',1)[0].rsplit('-',1)[0]
    if uttid in uttlist_dev:
        fout_dev_handle.write(line)
    else:
        fout_train_handle.write(line)

fout_train_handle.close()
fout_dev_handle.close()
