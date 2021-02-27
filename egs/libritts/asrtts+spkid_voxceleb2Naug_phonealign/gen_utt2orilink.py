# This script generates uttid and arklink (arklink is a link to the original feature sequence before the noise is added. If uttid corresponds to an original feature already, arklink is None)
# Usages: gen_utt2orilink.py [feats.scp of the train & dev combined after feat. is normed] [utt2spk to filter out uttids for output] [output file]
# Example: gen_utt2orilink.py dump/voxceleb2Naug/feats.scp data/voxceleb2Naug_train/utt2spk dump/voxceleb2Naug_train/utt2orilink
# Output: [output file]

import sys
from collections import OrderedDict

fname_feat = sys.argv[1]
fname_utt2spk = sys.argv[2]
fname_utt2orilink = sys.argv[3]

noise = ['babble', 'music', 'noise', 'reverb']

# build dic (feat-link pairs)
utt2link = OrderedDict()
for line in open(fname_feat):
    uttid, link = line.strip().split()
    if not any([ntype in uttid for ntype in noise]):
        utt2link[uttid] = link

# generate uttid-orilink file
uttlist = []
for line in open(fname_utt2spk):
    uttlist.append(line.strip().split()[0])

fwrite = open(fname_utt2orilink, 'w')
for uttid in uttlist:
    if any([ntype in uttid for ntype in noise]):
        fwrite.write(uttid + ' ' + utt2link[uttid.rsplit('-',maxsplit=1)[0]] + '\n')
    else:
        fwrite.write(uttid + ' None\n')

fwrite.close()

