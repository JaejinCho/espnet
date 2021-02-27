# Description: Generate a new utt2spk from the original utt2spk based on a new segment file
# Usage: python thisscript.py [new seg file] [utt2spk] [new utt2spk]
# e.g.: python thisscript.py new_segments utt2spk new_utt2spk

import sys

f_seg_in = sys.argv[1] # a file path for a new segment file (after cat and del for some segments)
f_utt2spk_in = sys.argv[2] # utt2spk for the original (NOT after processed for segments)
f_utt2spk_out = sys.argv[3] # new utt2spk based on the new segment file

# build a dict from utt to spk
utt2spk = {}
for line in open(f_utt2spk_in):
    temp = line.strip().split()
    utt2spk[temp[0].strip()] = temp[1].strip()

print("# utts in the original utt2spk: {}".format(len(utt2spk)))


f_handle_utt2spk_out = open(f_utt2spk_out,'w')
for line in open(f_seg_in):
    temp = line.strip().split()
    spkid = utt2spk[temp[1].strip()] # uttid without segment's time marks
    segid = temp[0].strip()
    f_handle_utt2spk_out.write('{0} {1}\n'.format(segid, spkid))

f_handle_utt2spk_out.close()
