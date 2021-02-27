#!/usr/bin/env python3

# Edited from asrtts+spkid_voxceleb2Naug_phonealign/gen_utt2orilink.py
# This script generates uttid and arklink (arklink is a link to the original feature sequence before the noise is added. If uttid corresponds to an original feature already, arklink is None)
# Usages: uttid2clean_featark.py [input: feats.scp including both clean and noise-augmented uttid-featark pairs] [output: uttid2clean_featark including the same uttid list as feats.scp but uttid-clean_featark pairs]
# Example: uttid2clean_featark.py dump/[corpus name]/feats.scp dump/[corpus name]/uttid2clean_featark
# --- feats.scp ---
# utt1          utt1.ark
# utt1-noise    utt1-noise.ark
# --- utt2orilink ---
# utt1          None
# utt1-noise    utt1.ark

import sys

fname_feat = sys.argv[1]
fname_utt2clean_featark = sys.argv[2]

noise = ['babble', 'music', 'noise', 'reverb']

utt2clean_featark = {}
uttlist = []
for line in open(fname_feat):
    uttid, link = line.strip().split()
    uttlist.append(uttid)
    if not any([ntype in uttid for ntype in noise]):
        utt2clean_featark[uttid] = link

# generate uttid2clean_featark
fwrite = open(fname_utt2clean_featark, 'w')
for uttid in uttlist:
    if any([ntype in uttid for ntype in noise]):
        fwrite.write(uttid + ' ' + utt2clean_featark[uttid.rsplit('-',maxsplit=1)[0]] + '\n')
    else:
        fwrite.write(uttid + ' None\n')

fwrite.close()
