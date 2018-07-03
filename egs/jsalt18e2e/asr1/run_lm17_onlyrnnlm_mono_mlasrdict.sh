#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a baseline for "JSALT'18 Multilingual End-to-end ASR for Incomplete Data"
# We use 5 Babel language (Assamese Tagalog Swahili Lao Zulu), Librispeech (English), and CSJ (Japanese)
# as a target language, and use 10 Babel language (Cantonese Bengali Pashto Turkish Vietnamese
# Haitian Tamil Kurmanji Tok-Pisin Georgian) as a non-target language.
# The recipe first build language-independent ASR by using non-target languages

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=50
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# rnnlm related
lm_weight=0.4
lm_lang=

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# exp tag
tag="" # tag for managing experiments.

# data set
# non-target languages: cantonese bengali pashto turkish vietnamese haitian tamil kurmanji tokpisin georgian
train_set=tr_babel10
train_dev=dt_babel10
# non-target
recog_set="dt_babel_cantonese et_babel_cantonese dt_babel_bengali et_babel_bengali dt_babel_pashto et_babel_pashto dt_babel_turkish et_babel_turkish\
 dt_babel_vietnamese et_babel_vietnamese dt_babel_haitian et_babel_haitian\
 dt_babel_tamil et_babel_tamil dt_babel_kurmanji et_babel_kurmanji dt_babel_tokpisin et_babel_tokpisin dt_babel_georgian et_babel_georgian"
# target
recog_set="dt_babel_assamese et_babel_assamese dt_babel_tagalog et_babel_tagalog dt_babel_swahili et_babel_swahili dt_babel_lao et_babel_lao dt_babel_zulu et_babel_zulu
 dt_csj_japanese et_csj_japanese_1 et_csj_japanese_2 et_csj_japanese_3\
 dt_libri_english_clean dt_libri_english_other et_libri_english_clean et_libri_english_other"
# whole set
recog_set="dt_babel_cantonese et_babel_cantonese dt_babel_assamese et_babel_assamese dt_babel_bengali et_babel_bengali dt_babel_pashto et_babel_pashto dt_babel_turkish et_babel_turkish\
 dt_babel_vietnamese et_babel_vietnamese dt_babel_haitian et_babel_haitian dt_babel_swahili et_babel_swahili dt_babel_lao et_babel_lao dt_babel_tagalog et_babel_tagalog\
 dt_babel_tamil et_babel_tamil dt_babel_kurmanji et_babel_kurmanji dt_babel_zulu et_babel_zulu dt_babel_tokpisin et_babel_tokpisin dt_babel_georgian et_babel_georgian\
 dt_csj_japanese et_csj_japanese_1 et_csj_japanese_2 et_csj_japanese_3\
 dt_libri_english_clean dt_libri_english_other et_libri_english_clean et_libri_english_other"

. utils/parse_options.sh || exit 1;

# data directories
csjdir=../../csj
libridir=../../librispeech
babeldir=../../babel

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dict=data/lang_1char/train_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

# You can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs256_mono_mlasrdict_${lm_lang}
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_mono_${lm_lang}
    mkdir -p ${lmdatadir}
    cat data/tr_babel_${lm_lang}/text | text2token.py -s 1 -n 1 -l ${nlsyms} | \
        cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/train.txt
    cat data/dt_babel_${lm_lang}/text | text2token.py -s 1 -n 1 -l ${nlsyms} | \
        cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
        lmngpu=1
    else
        lmngpu=${ngpu}
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${lmngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --epoch 60 \
        --batchsize 256 \
        --dict ${dict}
fi
