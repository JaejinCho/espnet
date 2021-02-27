# THIS WILL GENERATE *uttlist + train_trainNdev.utt2spklab @ dname
import sys
import random
from collections import OrderedDict

dname = sys.argv[1] # input directory name that includes spk2utt file. uttlists will be stored here

# get actual spk2utt mapping
spk2utt = {}
for line in open(dname + '/spk2utt'):
    elements = line.split()
    sess_id = elements[0] # sess_id = ${spkid}_${sess_id}
    spkid = sess_id.split('_')[0]
    if spkid not in spk2utt:
        spk2utt[spkid] = elements[1:]
    else:
        spk2utt[spkid].extend(elements[1:])

assert len(spk2utt) == 1151 or len(spk2utt) == 1150, 'num utts should be 1151 or 1150 but got {}'.format(len(spk2utt))

# get actuall spk2num
spk2num = {}
for spk in spk2utt:
    spk2num[spk] = len(spk2utt[spk])
spk2num = OrderedDict(sorted(spk2num.items(), key = lambda x: x[-1])) # To make sure NOT to get utterances from a speaker having utt < 10 for train_{train,dev} set after 1k utts are filled up for the set

# separte speakers into train_train and train_eval to have NO overlapped
# speakers across the subsets
subset2spks = {'train_train':[], 'train_eval':[]} # train_train for speaker discrimitive train, train_eval for enrollment and test
cnt_less10 = 0
for spk in spk2num:
    if spk2num[spk] < 10:
        cnt_less10 += 1
        subset2spks['train_train'].append(spk)
    else:
        if random.random() < 1000/(1151-cnt_less10-50) and len(subset2spks['train_train']) < 1000:
            subset2spks['train_train'].append(spk)
        else:
            subset2spks['train_eval'].append(spk)

# write a file
f_handles = {'utt2spklab': open(dname + '/train_trainNdev.utt2spklab', 'w')}
f_handles['train_dev'] = open(dname + '/train_dev.uttlist', 'w')
for subset in subset2spks:
    f_handles[subset] = open(dname + '/' + subset + '.uttlist', 'w')
#utts for train_eval
for spk in subset2spks['train_eval']:
    f_handles['train_eval'].write('\n'.join(spk2utt[spk]) + '\n')
#utts for train_train and train_dev (speakers should overlap for discriminative training for speakers)
spk_idx=0
for spk in subset2spks['train_train']:
    if spk2num[spk] < 10:
        for utt in spk2utt[spk]:
            f_handles['train_train'].write(utt + '\n')
            f_handles['utt2spklab'].write(utt + ' {}\n'.format(spk_idx))
    else:
        for utt in spk2utt[spk]:
            if random.random() < 0.04:
                f_handles['train_dev'].write(utt + '\n')
                f_handles['utt2spklab'].write(utt + ' {}\n'.format(spk_idx))
            else:
                f_handles['train_train'].write(utt + '\n')
                f_handles['utt2spklab'].write(utt + ' {}\n'.format(spk_idx))
    spk_idx += 1
assert spk_idx == 1000, "num spk is not 1000 for spk discrimitive train but {}".format(spk_idx)

# close file handles
for subset in f_handles:
    f_handles[subset].close()
