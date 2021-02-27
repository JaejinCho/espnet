# Gen segments_[short utt length in frame]*step_size to the dir where utt2num_frames is
# Usage: python gen_shortutt_seg.py [utt2num_frames] [short utt length in frame]
# e.g.: python gen_shortutt_seg.py dump/voxceleb1_test/utt2num_frames 300
# (TODO) Some parts to check if the step size is 10 msec (currently what this
# script assumes)
# (TODO) Currently we assume short utt length as int

import sys
import os
import random

# function
def get_chunk_dur(uttlen, max_chunk_len):
    """
    Get a random part corresponds to max_chunk_len
    - input:
        (int) uttlen: uttlen in frames
        (int) max_chunk_len: max_chunk_len in frames
    - output:
        (int) start: start point in frame
        (int) end: end point in frame
    """
    max_start = uttlen - max_chunk_len + 1
    start = random.randint(1,max_start)
    end = start + max_chunk_len - 1
    return start, end

# main
step_size=0.01 # in sec ((TODO) this will be calculated later as n_shift/fs)
fname_utt2num_frames = sys.argv[1]
max_chunk_len = int(sys.argv[2]) # in frames
fhandle_segments = open(os.path.dirname(fname_utt2num_frames) + '/segments_' + str(max_chunk_len), 'w')

for line in open(fname_utt2num_frames):
    uttid, uttlen = line.strip().split()
    if int(uttlen) > max_chunk_len:
        start, end = get_chunk_dur(int(uttlen), max_chunk_len)
        fhandle_segments.write(uttid + '-seg' + str(max_chunk_len) + ' ' + uttid + ' ' + str(start * step_size) + ' ' + str(end * step_size) + '\n')
    else:
        print("uttlen: {} =< max_chunk_len: {}".format(int(uttlen),max_chunk_len))
        start = 1
        end = int(uttlen)
        fhandle_segments.write(uttid + ' ' + uttid + ' ' + str(start * step_size) + ' ' + str(end * step_size) + '\n')

fhandle_segments.close()

