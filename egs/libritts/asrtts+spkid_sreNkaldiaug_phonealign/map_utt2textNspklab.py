#!/usr/bin/env python3

# e.g., map_utt2textNspklab.py ${sre_dir}/data/${cname}/ data/${cname}_aug_1m_${subset}/
# This will write a "text" and "utt2spklab" files under data/${cname}_aug_1m_${subset}/ based on ${sre_dir}/data/${cname}/
import sys

dpath_whole = sys.argv[1]
dname_aug = sys.argv[2]

# get utt2text mapping from the whole portion of a given corpus
utt2text_whole = {}
#for line in open(dpath_whole + '/text'): # error "UnicodeDecodeError: 'ascii' codec can't decode byte 0xe0 in position 274: ordinal not in range(128)"
for line in open(dpath_whole + '/text', 'r', encoding="utf-8"):
    uttid, text = line.strip().split(' ',1)
    utt2text_whole[uttid] = text

# get utt2spklab mapping from the whole portion of a given corpus
utt2spklab_whole = {}
for line in open(dpath_whole + '/utt2spklab'):
    uttid, spklab = line.strip().split(' ',1)
    utt2spklab_whole[uttid] = spklab

# write to file
fout_text_handle = open(dname_aug + '/text', 'w', encoding="utf-8")
fout_utt2spklab_handle = open(dname_aug + '/utt2spklab', 'w')
for line in open(dname_aug + '/utt2spk'):
    uttid_raw = line.strip().split(' ',1)[0] # e.g., utt1-noise
    uttid = uttid_raw.rsplit('-',1)[0] # e.g., utt1
    fout_text_handle.write("{0} {1}\n".format(uttid_raw,utt2text_whole[uttid]))
    fout_utt2spklab_handle.write("{0} {1}\n".format(uttid_raw,utt2spklab_whole[uttid]))

fout_text_handle.close()
fout_utt2spklab_handle.close()
