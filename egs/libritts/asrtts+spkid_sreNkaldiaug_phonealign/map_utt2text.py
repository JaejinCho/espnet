#!/usr/bin/env python3

# e.g., map_utt2text.py ${sre_dir}/data/${cname}/ data/${cname}_aug_1m/
# This will write a "text" file under data/${cname}_aug_1m/ based on ${sre_dir}/data/${cname}/text
import sys

dpath_whole = sys.argv[1]
dname_aug = sys.argv[2]

# get utt2text mapping from the whole portion of a given corpus
utt2text_whole = {}
for line in open(dpath_whole + '/text', 'r', encoding="utf-8"):
    uttid, text = line.strip().split(' ',1)
    utt2text_whole[uttid] = text

# write to file
fout_text_handle = open(dname_aug + '/text', 'w', encoding="utf-8")
for line in open(dname_aug + '/utt2spk'):
    uttid_raw = line.strip().split(' ',1)[0] # e.g., utt1-noise
    uttid = uttid_raw.rsplit('-',1)[0] # e.g., utt1
    fout_text_handle.write("{0} {1}\n".format(uttid_raw,utt2text_whole[uttid]))

fout_text_handle.close()
