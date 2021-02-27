import sys
import numpy as np

f_utt2num_frames=sys.argv[1]

list_num_frames = []
for line in open(f_utt2num_frames):
    list_num_frames.append(int(line.strip().split()[-1]))

# stats
print("min: {}".format(np.min(list_num_frames)))
print("max: {}".format(np.max(list_num_frames)))
print("mean: {}".format(np.mean(list_num_frames)))
print("std: {}".format(np.std(list_num_frames)))
