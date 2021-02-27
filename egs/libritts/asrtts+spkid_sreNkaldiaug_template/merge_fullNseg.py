# Use py3 to run this
# Description: To replace whole-utt embeddings by the original whole-utt embeddings + add the original whole-utt
# Usage: python merge_fullNseg.py [full emb fname (in)] [seg emb fname (in)] [new combined merged emb (out)]
import sys
import numpy as np
import re

fname_fullemb = sys.argv[1]
fname_segemb = sys.argv[2]
fname_outemb = sys.argv[3]

# 1. Get dictionary from a fullemb file
utt2fullemb = {}
for line in open(fname_fullemb):
    temp = line.strip().split()
    uttid = temp[0]
    feat = np.array(temp[2:-1], dtype=float)
    utt2fullemb[uttid] = feat

# 2. Write a new embed
fhand_write = open(fname_outemb,'w')
cnt_fullemb = 0
cnt_segemb = 0
for line in open(fname_segemb):
    temp = line.strip().split()
    uttid = temp[0]
    emb = np.array(temp[2:-1], dtype=float)
    # write here
    match = re.search('-seg\d\d\d', uttid) # reference for re.search: https://docs.python.org/3/library/re.html#re.search
    if match:
        # 1) write fullemb
        oriuttid = uttid[:match.start()]
        oriemb = utt2fullemb[oriuttid]
        fhand_write.write(oriuttid + ' [ ' + ' '.join(map(str,oriemb)) + ' ]\n')
        cnt_fullemb += 1
        # 2) write segemb
        fhand_write.write(uttid + ' [ ' + ' '.join(map(str,emb)) + ' ]\n')
        cnt_segemb += 1
    else: # write only fullemb (replace). For the reason of the replacement, refer to /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/list_run.debug.emb.log
        oriemb = utt2fullemb[uttid]
        fhand_write.write(uttid + ' [ ' + ' '.join(map(str,oriemb)) + ' ]\n')
        cnt_fullemb += 1

# 3. Finalize
fhand_write.close()
print('# ori fullemb: {}'.format(len(utt2fullemb)))
print('# fullemb (added + replaced): {}'.format(cnt_fullemb))
print('# segemb (after full segembs are removed): {}'.format(cnt_segemb))
print('# total mixed (full + seg) emb: {}. This should be the same as # lines in a new file'.format(cnt_segemb+cnt_fullemb))

