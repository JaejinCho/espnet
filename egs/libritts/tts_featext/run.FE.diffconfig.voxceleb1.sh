#!/bin/bash

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

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    if [ ! -d ${train_set_ori} ]; then
        echo "Generate ${train_set_ori} directory first" && exit 1
    fi
    if [ ! -d data/voxceleb1_train_filtered ];then
        mkdir -p data && cp -r ${train_set_ori} data/voxceleb1_train_filtered
    fi
    if [ ! -d data/${eval_set} ]; then
        cp -r /export/b11/jcho/kaldi_20190316/egs/librispeech/s5/data/voxceleb1_test data/voxceleb1_test
    fi

    # get subsets by uttlist files
    for fpath in `ls /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/data/voxceleb1_train_filtered/voxceleb1_train_filtered_{train,dev}.uttlist`;do
        dname=$(basename ${fpath} | cut -d'.' -f1) # this is same as dname=$(basename ${fpath%.*})
        subset_data_dir.sh --utt-list ${fpath} data/voxceleb1_train_filtered/ data/${dname}/
        ls -A data/${dname}/*
    done
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
utils/utt2spk_to_spk2utt.pl data/${eval_set}/utt2spk > data/${eval_set}/spk2utt
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation (used espnet version NOT kaldi to be consistent with wavform synthesis later using griffi_lim"

    fbankdir=fbank
    for x in ${train_set} ${dev_set} ${eval_set};do
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${x} \
            exp/make_fbank/${x} \
            ${fbankdir}
    done

    utils/combine_data.sh data/${train_set}_org data/${train_set}
    utils/combine_data.sh data/${dev_set}_org data/${dev_set}

    # remove utt having more than 3000 frames (only consider --maxframes since
    # input and # output frames are matching or anyway related with a
    # consistent factor (e.g. by frame_subsamling_factor in kaldi asr)
    remove_longshortdata_onlybymaxframes.sh --maxframes 3000 data/${train_set}_org data/${train_set}
    remove_longshortdata_onlybymaxframes.sh --maxframes 3000 data/${dev_set}_org data/${dev_set}

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi
