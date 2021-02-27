#!/usr/bin/env python3
# Edited from average_xvector.py

# Use this script with python3
# Description: This script extracts one xvector per utt from the utt segments
# Usage: python extract_uttxvector_fromsegs.py (--random [utt2num_frames]) --model [model_path] [in_feat_scp] [out_emb_arktext]
# (TODO: Should I write a file for random selection of segments?)
# (TODO: There are multiple ways that I can do when I get random chunks)

import numpy as np
import random
import argparse
import torch
import kaldiio
import logging
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.asr.asr_utils import get_model_conf
from espnet.utils.dynamic_import import dynamic_import
from espnet.nets.tts_interface import TTSInterface
from espnet.asr.asr_utils import torch_load


parser = argparse.ArgumentParser(description='Write an xvector per utt from concated segment-wise feature sequences')
parser.add_argument('fname_in_feat', type=str,
                    help='input: file path for segment-wise features in kaldi scp format')
parser.add_argument('fname_out_emb', type=str,
                    help='output: final utt-wise embedding file name')
parser.add_argument('--random', type=str, dest='fname_in_numframes', default=None,
                    help='utt2num_frames file name if we want to do random segments selection before segments concatenation. Currently total # frames from the random segments ranges between 1k and 6k in frames')
# to read a trained model related arguments (trimmed from espnet/bin/speakerid_decode.py argparse)
parser.add_argument('--model', type=str, required=True,
                    help='Model file parameters to read')
parser.add_argument('--model-conf', type=str, default=None,
                    help='Model config file')
parser.add_argument('--ngpu', default=0, type=int,
                    help='Number of GPUs')
parser.add_argument('--seed', default=1, type=int,
                    help='Random seed')
parser.add_argument('--debugmode', default=1, type=int,
                    help='Debugmode')
                    

args = parser.parse_args()

random_seed = 1 # this is for deterministic result when using "shuffle" below
random.seed(random_seed) # for randint below to get deterministic result (i.e., the same results from running this script with same data)

def read_model(args):
    """Decode with E2E-TTS model."""
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info('args: ' + key + ': ' + str(vars(args)[key]))

    # define model
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, TTSInterface)
    logging.info(model)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    torch_load(args.model, model)
    model.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)
    
    return model.resnet_spkid

def process_segs2uttxvec(accu_uttwise, uttid_prev, fhandle_write, total_nframes=None, max_seg_nframes=1000, spk_enc=None):
    '''
    This writes one x-vector from a given list, accu_uttwise. (Optionally with random segments selection)
    accu_uttwise (list): its element is a list (int: seg_nframes, np.array: seg_featseq) or a np.array: seg_featseq
    uttid_prev (str): uttid to be used with the resulting utt-wise xvector in writing an xvector file
    fhandle_write (file handle): file hand to write an xvector file
    total_nframes (int): Only when selecting random segments before segments concatenation, a random int given to upperbound the number of total frames accumulated over the selected segments
    max_seg_nframes (int): Only when selecting random segments before segments concatenation, if a current segment is longer than this value in frame, I will move the segment to the end and use it later if still in while loop after processing others
    spk_enc : Speaker encoder to encode feature seqs to an x-vector
    '''
    #sum_xvec = 0
    acc_feat= []
    accu_nframes = 0
    cnt_seg = 0
    if total_nframes: # concatenation with random segments selection
        # Shuffle segments and accumulate them for concatenation until accumulated # frames (accu_nframes) gets to total_nframes. Set random_seed in the beginning for shuffle for deterministic results
        random.Random(random_seed).shuffle(accu_uttwise)
        while (len(accu_uttwise) != 0) and (accu_nframes < total_nframes):
            seg_nframes = accu_uttwise[0][0]
            if seg_nframes < 0: # this is when left-rotated element gets back to the front again
                acc_feat.append(accu_uttwise[0][1])
                accu_nframes -= seg_nframes
                cnt_seg += 1
                del accu_uttwise[0]
            elif seg_nframes > max_seg_nframes:
                accu_uttwise[0][0] = -seg_nframes # this is to mark that this element left-rotated once
                accu_uttwise = accu_uttwise[1:] + accu_uttwise[:1] # left rotate once
            else:
                acc_feat.append(accu_uttwise[0][1])
                accu_nframes += seg_nframes
                cnt_seg += 1
                del accu_uttwise[0]
    else: # accumulation all segments for concatenation
        while (len(accu_uttwise) != 0):
            acc_feat.append(accu_uttwise[0])
            cnt_seg += 1
            del accu_uttwise[0]

    y = [np.concatenate(acc_feat,axis=0)]
    device=next(spk_enc.parameters()).device
    utt_xvec = spk_enc.predict(torch.from_numpy(np.array(y,dtype=np.float32)).to(device)).cpu().data.numpy().flatten() # (TODO: get the "device" value by checking the model (idk if it is possible))
    fhandle_write.write(uttid_prev + ' [ '+' '.join(map(str, utt_xvec.tolist()))+' ]\n')

##### main #####
uttid_prev = ''
accu_uttwise = [] # this accumulates feature location and (optionally, its num_frames) over segments in one utt 
fhandle_write = open(args.fname_out_emb,'w')
spk_enc = read_model(args)

if args.fname_in_numframes:
    # case 1: utt2num_frames are given too (select segments and only concatenate feature sequences only from the selected segments)
    with open(args.fname_in_feat) as f_in_feat, open(args.fname_in_numframes) as f_in_nframe:
        for line1, line2 in zip(f_in_feat, f_in_nframe):
            segid1, seg_featpos = line1.strip().split(maxsplit=1)
            segid2, seg_nframes = line2.strip().split(maxsplit=1)
            assert segid1 == segid2, "emb and utt2num_frames files were NOT sorted in the same way"
            uttid_cur = segid1.rsplit(sep='-',maxsplit=1)[0]
            if uttid_cur == uttid_prev:
                accu_uttwise.append([int(seg_nframes), kaldiio.load_mat(seg_featpos)])
            else:
                if len(accu_uttwise) != 0: # to filter out very first line
                    process_segs2uttxvec(accu_uttwise, uttid_prev, fhandle_write, total_nframes=random.randint(1000,6000), spk_enc=spk_enc)
                accu_uttwise = [[int(seg_nframes), kaldiio.load_mat(seg_featpos)]] # initialize
            uttid_prev = uttid_cur
    
    # For the last uttid, it came out the from block above W/O "process_segs2uttxvec" applied to the accu_uttwise
    process_segs2uttxvec(accu_uttwise, uttid_prev, fhandle_write, total_nframes=random.randint(1000,6000), spk_enc=spk_enc)

else:
    # case 2: utt2num_frames are NOT given (simply concatenate all segment-wise feature sequences)
    for line in open(args.fname_in_feat):
        segid, seg_featpos = line.strip().split(maxsplit=1)
        uttid_cur = segid.rsplit(sep='-',maxsplit=1)[0]
        if uttid_cur == uttid_prev:
            accu_uttwise.append(kaldiio.load_mat(seg_featpos))
        else:
            if len(accu_uttwise) != 0: # to filter out very first line
                process_segs2uttxvec(accu_uttwise, uttid_prev, fhandle_write, spk_enc=spk_enc)
            accu_uttwise = [kaldiio.load_mat(seg_featpos)] # initialize
        uttid_prev = uttid_cur
    
    # For the last uttid, it came out the with block above W/O "process_segs2uttxvec" for the accu_uttwise
    process_segs2uttxvec(accu_uttwise, uttid_prev, fhandle_write, spk_enc=spk_enc)

fhandle_write.close()
