#!/usr/bin/env python3

# Edited from separate_trainNdev_uttlists.py
# Edited from /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/make_uttlists_voxceleb1_filtered.py
# Usage: thisscript.py dname_ori dname_train dname_dev percent_dev
# Input argument:
# - dname_ori: original directory name before separation
# - dname_train: train_directory name
# - dname_dev: dev_directory name
# - percent_dev: how much portion will be dev set
# Output (at data/${dname_ori}):
# - ${dname_train}.uttlist
# - ${dname_dev}.uttlist
# - ${dname_ori}.utt2spklab
# e.g) separate_trainNdev_uttlists.py voxceleb_train ${train_set} ${dev_set} 0.04 ${start_spkidx} # where train_set=voxceleb_train_train, dev_set=voxceleb_train_dev

import sys
import random

dname_ori = sys.argv[1]
dname_train = sys.argv[2]
dname_dev = sys.argv[3]
percent_dev = float(sys.argv[4])
start_spkidx = int(sys.argv[5].strip())

dname = "./data/" + dname_ori + "/" # input directory name that includes spk2utt file. output files will be stored here

# spk to utt mapping (this is not necessary but for double-check)
spk2utt = {}
for line in open(dname + '/spk2utt'):
    elements = line.strip().split()
    spkid = elements[0]
    if spkid not in spk2utt:
        spk2utt[spkid] = elements[1:]
    else:
        print("WARNING: repeated spkid. adding the utts to the spk.")
        spk2utt[spkid].extend(elements[1:])

print("# spks is {}".format(len(spk2utt)))
print("# start_spkidx for the next corpus: {}".format(len(spk2utt) + start_spkidx)) # grep this and give it for the next corpus

# write on files
f_handles = {'utt2spklab': open(dname + '/utt2spklab', 'w')}
f_handles[dname_train] = open(dname + '/' + dname_train + '.uttlist' , 'w') # for training
f_handles[dname_dev] = open(dname + '/' + dname_dev + '.uttlist' , 'w') # for stopping training

random.seed(1)
for spk_idx, spk in enumerate(spk2utt):
    temp_utts = spk2utt[spk]
    random.shuffle(temp_utts)
    num_utts = len(temp_utts)
    num_dev = int(round(num_utts * percent_dev)) # the number of utterances for dev
    if num_dev < 1:
        print("WARNING: # utterances for devset is 0")
    for ix in range(num_dev): # for dev
        f_handles[dname_dev].write(temp_utts[ix] + '\n')
        f_handles['utt2spklab'].write(temp_utts[ix] + ' {}\n'.format(start_spkidx + spk_idx))
    for ix in range(num_dev,num_utts): # for train
        f_handles[dname_train].write(temp_utts[ix] + '\n')
        f_handles['utt2spklab'].write(temp_utts[ix] + ' {}\n'.format(start_spkidx + spk_idx))

for subset in f_handles:
    f_handles[subset].close()
