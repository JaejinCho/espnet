# This script is to compose trials (and divide the train_eval subset into
# enrollment and test subsets from the trials)
# e.g.) python thisscript.py data/train_eval/spk2gender data/train_eval/utt2spk
import sys
import os

fname_spk2gender = sys.argv[1]
fname_utt2spk = sys.argv[2]

# dictionary from spk to gender
spk2gender = {}
for line in open(fname_spk2gender):
    spk, gen = line.split()
    spk2gender[spk] = gen

# Get the upper triangular part from a square matrix composed from all the utterances
utts = [ line.split()[0] for line in map(str.strip,open(fname_utt2spk).readlines()) ]

ftrial = open(os.path.dirname(fname_utt2spk) + '/trials', 'w')
n_utts = len(utts)
cnt_valid = 0
cnt_crossgender = 0
for i in range(n_utts):
    for j in range(i+1,n_utts):
        spk_enroll = '_'.join(utts[i].split('_')[:2])
        spk_test = '_'.join(utts[j].split('_')[:2])
        if spk2gender[spk_enroll] != spk2gender[spk_test]:
            cnt_crossgender += 1
            continue
        else:
            if spk_enroll == spk_test:
                ftrial.write(utts[i] + ' ' + utts[j] + ' target\n')
            else:
                ftrial.write(utts[i] + ' ' + utts[j] + ' nontarget\n')
            cnt_valid += 1

assert cnt_valid == (n_utts**2 - n_utts)/2 - cnt_crossgender, "Error: cnt {}, # pairs {}".format(cnt_valid, (n_utts**2 - n_utts)/2 - cnt_crossgender)

ftrial.close()

