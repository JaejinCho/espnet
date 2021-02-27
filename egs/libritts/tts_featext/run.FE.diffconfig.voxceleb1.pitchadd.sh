#!/bin/bash

# (JJ) Edited from run.FE.diffconfig.voxceleb1.sh assuming
# run.FE.diffconfig.voxceleb1.sh was run already. Just add pitch with this
# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpu in training
nj=64        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points (~32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=160   # number of shift points (~10ms)
win_length="" # window length

# config files
train_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/60

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set_ori=/export/b11/jcho/kaldi_20190316/egs/librispeech/s5/data/voxceleb1_train_filtered
train_set=voxceleb1_train_filtered_train
dev_set=voxceleb1_train_filtered_dev
eval_set=voxceleb1_test

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Extract pitch"

    pitchdir=pitch
    for x in ${train_set} ${dev_set} ${eval_set};do
        # (JJ) wrote steps/make_pitch.sh
        calc(){ awk "BEGIN { print "$*" }"; }
        frame_len=`calc ${n_fft}/${fs}*1000` # cal. frame_len in ms

        steps/make_pitch.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} --frame_length ${frame_len} \
            data/${x} \
            exp/make_pitch/${x} \
            ${pitchdir}
    done
fi
