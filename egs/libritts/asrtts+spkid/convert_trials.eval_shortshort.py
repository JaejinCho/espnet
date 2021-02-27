# (JJ) Copied and edited from ../asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/
# This converts a original trials to new trials for short-segment eval.
# Usage: python convert_trials.py [ori utt2num_frames in] [seg utt2num_frames in] [ori trials in] [seg trials out]
# Example: python convert_trials.py data/voxceleb1_test/utt2num_frames dump/voxceleb1_test_seg500/utt2num_frames data/voxceleb1_test/trials dump/voxceleb1_test_seg500/trials

import sys

fname_utt2num_frames_ori = sys.argv[1]
fname_utt2num_frames_seg = sys.argv[2]
fname_trials_ori = sys.argv[3]
fname_trials_seg = sys.argv[4]

# 1st: Create a dictionary from ori to seg uttid
ori2seg = {}
for line1, line2 in zip(open(fname_utt2num_frames_ori),open(fname_utt2num_frames_seg)):
    uttid1 = line1.strip().split()[0]
    uttid2 = line2.strip().split()[0]
    if uttid1 != uttid2:
        ori2seg[uttid1] = uttid2

# 2st: Read a file as one line and replace ori to uttid by running through all the key-value pairs
## (TODO) Change this part to be more exact next time
wholefile=open(fname_trials_ori).read()
fhandle_trials_seg=open(fname_trials_seg,'w')
for uttid1 in ori2seg:
    wholefile = wholefile.replace(uttid1,ori2seg[uttid1])

fhandle_trials_seg.write(wholefile)
fhandle_trials_seg.close()
