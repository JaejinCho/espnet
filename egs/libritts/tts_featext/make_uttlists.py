# Copied and edited from
# /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/make_uttlists_voxceleb1_filtered.py
# THIS WILL GENERATE ${basename}_{train,dev}.uttlist + ${basename}.utt2spklab @ ${dname} data dir
# Usage: python make_uttlists.py [datadir] [data portion for dev]
# e.g: python make_uttlists.py ./data/voxceleb_train_combined/ 0.03
import random
import sys
import re

dname =  sys.argv[1] # "./data/voxceleb_train_combined/" # input directory name that includes spk2utt file. output files will be stored here
percent_dev = float(sys.argv[2]) # $(percent_dev)% of utts assigned for model selection

temp = re.split('/+',dname)
if temp[-1] == '':
    basename = temp[-2]
else:
    basename = temp[-1]

print('Data dir name: {}'.format(basename))

# spk to utt mapping (this is not necessary but for double-check)
spk2utt = {}
for line in open(dname + '/spk2utt'):
    elements = line.strip().split()
    spkid = elements[0]
    if spkid not in spk2utt:
        spk2utt[spkid] = elements[1:]
    else:
        print("repeated spkid: adding the utts to the spk")
        spk2utt[spkid].extend(elements[1:])

print('Num spks is {}'.format(len(spk2utt)))

# write on files
f_handles = {'utt2spklab_filtered': open(dname + '/' + basename + '.utt2spklab', 'w')}
f_handles[ basename + '_train'] = open(dname + '/' + basename + '_train.uttlist', 'w') # for training
f_handles[ basename + '_dev'] = open(dname + '/' + basename + '_dev.uttlist', 'w') # for stopping trainning

random.seed(1)
for spk_idx, spk in enumerate(spk2utt):
    temp_utts = spk2utt[spk]
    random.shuffle(temp_utts)
    num_utts = len(temp_utts)
    num_dev = int(round(num_utts * percent_dev)) # the number of utterances for dev
    for ix in range(num_dev): # for dev
        f_handles[ basename + '_dev'].write(temp_utts[ix] + '\n')
        f_handles['utt2spklab_filtered'].write(temp_utts[ix] + ' {}\n'.format(spk_idx))
    for ix in range(num_dev,num_utts): # for train
        f_handles[ basename + '_train'].write(temp_utts[ix] + '\n')
        f_handles['utt2spklab_filtered'].write(temp_utts[ix] + ' {}\n'.format(spk_idx))

for subset in f_handles:
    f_handles[subset].close()
