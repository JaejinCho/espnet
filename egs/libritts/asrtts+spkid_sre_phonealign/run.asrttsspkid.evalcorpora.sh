#!/bin/bash

# Edited from run.asrttsspkid.sh
# JJ: Copied and edited from
# ../asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/run.asrttsspkid.spkloss_weight.new.update.rf3.sh.
# v1.
# Only change the data prep. This requires to expand text also for augmented data
# v2. (TODO in another script)
# (Najim) Train speaker ID only with noisy and train TTS + SpkID with clean

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0
stop_stage=100
ngpu=1       # number of gpu in training
nj=64        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)
cuda_memsize=50

# feature extraction related (set by following /export/b18/jcho/espnet3/egs/libritts/tts_featext/run.FE.diffconfig.voxceleb1.sh)
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points (32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=160   # number of shift points (10ms)
win_length="" # window length

# config files
train_config=conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Speaker id related
spkloss_weight=0.03

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

feat_dir=/export/b18/jcho/espnet3/egs/libritts/tts_featext
trans_dir=/export/b11/jcho/kaldi_20190316/egs/babel/masr_grapheme/exp/chain/tdnn1h_d_sp/
train_set=sre20_train
#train_set=voxceleb2_train
#dev_set=voxceleb2_dev
#eval_set=voxceleb1_test

corpora_list="sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test"

# stage 0 and 1 in one
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    if [ ! -d data ]; then
        mkdir -p data
    fi
    # 0. Copy needed data
    ## data dir w/o text (each corpus has "feats.scp  filetype  segments  spk2utt  utt2num_frames  utt2spk  wav.scp")
    for cname in ${corpora_list}; do
        # cp data dir & create the dump dir
        if [ ! -d data/${cname} ]; then
            cp -r ${feat_dir}/data/${cname} data/
            mkdir -p ${dumpdir}/${cname}_eval
        fi
    done

    # 1. Feature normalizaiton
    echo "stage 1: Feature Generation (dump using apply-cmvn-sliding)"
    pids=()
    for cname in ${corpora_list}; do
        dump_cmvnsliding.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${cname}/feats.scp exp/dump_feats/${cname}_eval ${dumpdir}/${cname}_eval &
        pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
