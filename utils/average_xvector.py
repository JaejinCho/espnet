#!/usr/bin/env python3

# Use this script with python3
# Description: This script averages x-vectors
# Usage: python average_xvector.py (--random [utt2num_frames]) [in_emb_arktext] [out_emb_arktext]
# (TODO: Should I write a file for which parts I averaged?)
# (TODO: There are multiple ways that I can do when I get random chunks)

import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='Write to a file an averged xvector over the segment-wise x-vectors')
parser.add_argument('fname_in_emb', type=str,
                    help='input: segment-wise embdding file name')
parser.add_argument('fname_out_emb', type=str,
                    help='output: averaged utt-wise embedding file name')
parser.add_argument('--random', type=str, dest='fname_in_numframes', default=None,
                    help='utt2num_frames file name if we want to do random segments selection before average. Currently total # frames from the random segments ranges between 1k and 6k in frames')
args = parser.parse_args()

random_seed = 1 # this is for deterministic result when using "shuffle" below
random.seed(random_seed) # for randint below to get deterministic result (i.e., the same results from running this script with same data)

def convert_kaldivec2nparray(kaldivec):
    '''
    It converts the string "[ 1 2 3 ... ]" in kaldivec to np.array
    kaldivec (str): a vector in str format from kaldi xvetor.ark
    '''
    return np.array(kaldivec[1:-1].strip().split(), dtype=float)

def process_average(accu_uttwise, uttid_prev, fhandle_write, total_nframes=None, max_seg_nframes=1000):
    '''
    This writes to a file (fhandle_write) one averaged x-vectors from a given list, accu_uttwise. (Optionally with random segments selection)
    accu_uttwise (list): its element is a list (int: seg_nframes, np.array: seg_xvec) or a np.array: seg_xvec
    uttid_prev (str): uttid to be used with the resulting averaged xvector in writing an xvector file
    fhandle_write (file handle): file hand to write an xvector file
    total_nframes (int): Only when selecting random segments before average, a random int given to upperbound the number of total frames accumulated over the selected segments
    max_seg_nframes (int): Only when selecting random segments before average, if a current segment is longer than this value in frame, I will move the segment to the end and use it later if still in while loop after processing others
    '''
    sum_xvec = 0
    accu_nframes = 0
    cnt_seg = 0
    if total_nframes: # average with random segments selection
        # Shuffle segments and average until accumulated # frames (accu_nframes) gets to total_nframes. Set random_seed in the beginning for shuffle for deterministic results
        random.Random(random_seed).shuffle(accu_uttwise)
        while (len(accu_uttwise) != 0) and (accu_nframes < total_nframes):
            seg_nframes = accu_uttwise[0][0]
            if seg_nframes < 0: # this is when left-rotated element gets back to the front again
                sum_xvec += accu_uttwise[0][1]
                accu_nframes -= seg_nframes
                cnt_seg += 1
                del accu_uttwise[0]
            elif seg_nframes > max_seg_nframes:
                accu_uttwise[0][0] = -seg_nframes # this is to mark that this element left-rotated once
                accu_uttwise = accu_uttwise[1:] + accu_uttwise[:1] # left rotate once
            else:
                sum_xvec += accu_uttwise[0][1]
                accu_nframes += seg_nframes
                cnt_seg += 1
                del accu_uttwise[0]
        # (TODO-2 for update) 1) Calculate the average num_frames, 2) sort the list of 2-element lists by the difference from average: with the absolute value within avg std range (TODO: calculate the std range), and without absolute out of the range & first with utts with more frames then, the utts with less frames
    else: # average over all segments
        while (len(accu_uttwise) != 0):
            sum_xvec += accu_uttwise[0]
            cnt_seg += 1
            del accu_uttwise[0]

    avg_xvec = sum_xvec/cnt_seg
    fhandle_write.write(uttid_prev + ' [ '+' '.join(map(str, avg_xvec.tolist()))+' ]\n')

##### main #####
uttid_prev = ''
accu_uttwise = [] # this accumulates x-vectors and its num_frames over segments in one utt
fhandle_write = open(args.fname_out_emb,'w')

if args.fname_in_numframes:
    # case 1: utt2num_frames are given too (select segments and only average over the selected segment-wise x-vectors)
    with open(args.fname_in_emb) as f_in_emb, open(args.fname_in_numframes) as f_in_nframe:
        for line1, line2 in zip(f_in_emb, f_in_nframe):
            segid1, seg_xvec = line1.strip().split(maxsplit=1)
            segid2, seg_nframes = line2.strip().split(maxsplit=1)
            assert segid1 == segid2, "emb and utt2num_frames files were NOT sorted in the same way"
            uttid_cur = segid1.rsplit(sep='-',maxsplit=1)[0]
            if uttid_cur == uttid_prev:
                accu_uttwise.append([int(seg_nframes), convert_kaldivec2nparray(seg_xvec)])
            else:
                if len(accu_uttwise) != 0: # to filter out very first line
                    process_average(accu_uttwise, uttid_prev, fhandle_write, total_nframes=random.randint(1000,6000))
                accu_uttwise = [[int(seg_nframes), convert_kaldivec2nparray(seg_xvec)]] # initialize
            uttid_prev = uttid_cur
    
    # For the last uttid, it came out the with block above W/O "process_average" for the accu_uttwise
    process_average(accu_uttwise, uttid_prev, fhandle_write, total_nframes=random.randint(1000,6000))

else:
    # case 2: utt2num_frames are NOT given (simply average over all segment-wise x-vectors)
    for line in open(args.fname_in_emb):
        segid, seg_xvec = line.strip().split(maxsplit=1)
        uttid_cur = segid.rsplit(sep='-',maxsplit=1)[0]
        if uttid_cur == uttid_prev:
            accu_uttwise.append(convert_kaldivec2nparray(seg_xvec))
        else:
            if len(accu_uttwise) != 0: # to filter out very first line
                process_average(accu_uttwise, uttid_prev, fhandle_write)
            accu_uttwise = [convert_kaldivec2nparray(seg_xvec)] # initialize
        uttid_prev = uttid_cur
    
    # For the last uttid, it came out the with block above W/O "process_average" for the accu_uttwise
    process_average(accu_uttwise, uttid_prev, fhandle_write)

fhandle_write.close()
