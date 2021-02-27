# THIS WILL GENERATE voxceleb1_train_filtered_{train,dev}.uttlist + voxceleb1_train_filtered.utt2spklab @ dname (voxceleb1_train_filtered data dir)
# For voxceleb1_train_filtered, spk2num ranges from 37 to 976 (so do not need to check the speakers that have less than 10 utts)
import random

dname = "/export/b11/jcho/espnet/egs/wsj/asr1/data/voxceleb1_train_filtered/" # input directory name that includes spk2utt file. output files will be stored here

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

assert len(spk2utt) == 1211, 'num spks should be 1211 but got {}'.format(len(spk2utt))

# write on files
f_handles = {'utt2spklab_filtered': open(dname + '/voxceleb1_train_filtered.utt2spklab', 'w')}
f_handles['voxceleb1_train_filtered_train'] = open(dname + '/voxceleb1_train_filtered_train.uttlist', 'w') # for training
f_handles['voxceleb1_train_filtered_dev'] = open(dname + '/voxceleb1_train_filtered_dev.uttlist', 'w') # for stopping trainning

percent_dev = 0.04 # 4% of utts assigned for model selection
random.seed(1)
for spk_idx, spk in enumerate(spk2utt):
    temp_utts = spk2utt[spk]
    random.shuffle(temp_utts)
    num_utts = len(temp_utts)
    num_dev = int(round(num_utts * percent_dev)) # the number of utterances for dev
    for ix in range(num_dev): # for dev
        f_handles['voxceleb1_train_filtered_dev'].write(temp_utts[ix] + '\n')
        f_handles['utt2spklab_filtered'].write(temp_utts[ix] + ' {}\n'.format(spk_idx))
    for ix in range(num_dev,num_utts): # for train
        f_handles['voxceleb1_train_filtered_train'].write(temp_utts[ix] + '\n')
        f_handles['utt2spklab_filtered'].write(temp_utts[ix] + ' {}\n'.format(spk_idx))

for subset in f_handles:
    f_handles[subset].close()
