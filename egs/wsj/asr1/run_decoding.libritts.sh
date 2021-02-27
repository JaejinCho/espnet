#!/bin/bash

# JJ: copied from run_ruizhi_onCmachine.sh and modified for data prep (copying)
# and decoding
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
#. ./cmd_c.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# network architecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=4
eunits=320
eprojs=320
subsample=1_1_1_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=add
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.2

# label smoothing
lsm_type=unigram
lsm_weight=0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_batchsize=300    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=30
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# data
wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
#. ./cmd_c.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
#train_dev=test_dev93
#train_test=test_eval92
#recog_set="test_dev93 test_eval92"

## JJ added start
libritts_dir=/export/b18/jcho/espnet3/egs/libritts/tts+spkid_final
libritts_train_set=train_train
libritts_train_dev=train_dev
# move data
if [ ${stage} -le 0 ]; then
    echo "stage 0: Move data dirs from different corpus"
for subset in ${libritts_train_set} ${libritts_train_dev};do
    cp -r ${libritts_dir}/data/${subset} data/
done
fi

libritts_feat_tr_dir=${dumpdir}/${libritts_train_set}/delta${do_delta}; mkdir -p ${libritts_feat_tr_dir}
libritts_feat_dt_dir=${dumpdir}/${libritts_train_dev}/delta${do_delta}; mkdir -p ${libritts_feat_dt_dir}
# generate dump (fine to use human transcripts here but ignore them later)
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for subset in ${libritts_train_set} ${libritts_train_dev};do
        steps/make_fbank_pitch_libritts.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${subset} exp/make_fbank/${subset} ${fbankdir}
    done

    # compute global CMVN (already done)
    #compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${libritts_feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${libritts_train_set}/delta${do_delta}/storage \
        ${libritts_feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${libritts_feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${libritts_train_dev}/delta${do_delta}/storage \
        ${libritts_feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${libritts_train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/libritts_train ${libritts_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${libritts_train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/libritts_dev ${libritts_feat_dt_dir}
fi

# run decode
if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_onCmachine
    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}

if [ ${stage} -le 5 ]; then
    echo "stage 5: GPU decoding"
    for rtask in ${libritts_train_set} ${libritts_train_dev}; do
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}_gpudecoding
        if [ $use_wordlm = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        if [ $lm_weight == 0 ]; then
            recog_opts=""
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        #### use GPU for decoding
        if [ ${ngpu} -eq 0 ]; then
            echo "ngpu should not be 0 for gpu decoding." && exit 1
        fi
        ngpu=${ngpu}
        mkdir -p ${expdir}/${decode_dir}/log

        asr_recog.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json ${feat_recog_dir}/data.json \
        --result-label ${expdir}/${decode_dir}/data.json \
        --model ${expdir}/results/${recog_model}  \
        --beam-size ${beam_size} \
        --penalty ${penalty} \
        --maxlenratio ${maxlenratio} \
        --minlenratio ${minlenratio} \
        --ctc-weight ${ctc_weight} \
        --lm-weight ${lm_weight} \
        $recog_opts 2>&1 | tee ${expdir}/${decode_dir}/log/decode.log

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    done
    echo "Finished"
fi

## JJ added end
