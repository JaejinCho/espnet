#!/usr/bin/env python3
# e.g.) python thisscript.py ${test_data_dir}/utt2num_frames ${seg_dir}/segments
# 1. find a discarded utts by comparing uttids in ${seg_dir}/segments & uttids
# in ${test_data_dir}/utt2num_frames
# 2. For all the utts that were deleted, add them to ${seg_dir}/segments file

import sys

fin_utt2num_frames = sys.argv[1]
f_segments = sys.argv[2]

utt2num_frames = {}
for line in open(fin_utt2num_frames):
    uttid, num_frames = line.strip().split()
    utt2num_frames[uttid] = float(num_frames)

uttlist_from_segments = []
for line in open(f_segments):
    uttlist_from_segments.append(line.strip().split()[1])
uttlist_from_segments = set(uttlist_from_segments)

fhandle_write = open(f_segments,'a')
for item in utt2num_frames.items():
    if item[0] not in uttlist_from_segments:
        print('uttid: {0} is discarded from SAD process. Recovering its entire utt as a segment by addingn to {1}', item[0], f_segments)
        end_time = "{:.2f}".format((item[1]-2) * 0.01) # -2 to avoid a possible error
        fhandle_write.write("{0}-0000000-{1} {0} {2} {3}\n".format(item[0], str(item[1]-2).zfill(7), '0.00', end_time))
fhandle_write.close()




