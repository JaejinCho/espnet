#!/bin/bash

# JJ: Edited from extract_spkemb_cpuparallel.labcorpus.sh to cover evalcorpus (this includes FE before x-vec extraction)
# JJ: For ESPnet expriments, I can simply change expdir to a new one in input
# arguments and run this script as a whole
# (although we separte training run*.sh scripts into two:
# run.asrttsspkid.spkloss_weight.voxceleb1_filtered_bs48.sh and
# run.spkidonly.final.voxceleb1.sh depending on if it is tts+spkid loss (0 possible) or
# spkidonly, this backend script is unified).

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0
stop_stage=100
ngpu=0       # number of gpu in training
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
n_fft=512    # number of fft points
n_shift=160   # number of shift points
win_length="" # window length

# config files
#train_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best # or give just a specific snapshot (without any path info. e.g. snapshot.ep.6)
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Speaker id related
#spkloss_weight=0

# Speaker id backend related
expdir=""
lda_dim=150

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/60

# exp tag
tag="" # tag for managing experiments.

# corpus for spkemb extraction
corpus_name=""

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


feat_dir=/export/b18/jcho/espnet3/egs/libritts/tts_featext
trans_dir=/export/b11/jcho/kaldi_20190316/egs/babel/masr_grapheme/exp/chain/tdnn1h_d_sp/
train_set=sre20_train

#corpora_list="sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test"

# stage 0 and 1 in one
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation. Skip if it's already DONE"
    if [ ! -d data ]; then
        mkdir -p data
    fi
    # 0. cp data dir (TODO: I might change this by symlinking if the copied
    # data contents would never change in later processes)
    if [ ! -d data/${corpus_name} ]; then
        cp -r ${feat_dir}/data/${corpus_name} data/
    else
        echo "Skip stage 0..."
    fi

    # 1. feature normalizaiton
    echo "stage 1: Feature Generation (dump using apply-cmvn-sliding). Skip if it's already DONE"
    if [ ! -d ${dumpdir}/${corpus_name}_evaluation ]; then
        mkdir -p ${dumpdir}/${corpus_name}_evaluation
        dump_cmvnsliding.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
            data/${corpus_name}/feats.scp exp/dump_feats/${corpus_name}_evaluation ${dumpdir}/${corpus_name}_evaluation
    else
        echo "Skip stage 1..."
    fi
fi



# x-vector extraction
if [[ model.loss.best == ${model} ]]; then
    embname=`basename ${expdir}`
else
    embname=`basename ${expdir}`_${model}
fi

echo Embedding name: ${embname}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    pids=() # initialize pids
    # decode in parallel with cpus
    for extract_set in ${corpus_name}_evaluation;do
    (
        # split feats.scp for cpu-parallel processing
        featscp=dump/${extract_set}/feats.scp
        splitdir=dump/${extract_set}/feats_split${nj}/
        if [ ! -d ${splitdir} ]; then
            mkdir -p ${splitdir}
            split_scps=""
            for n in $(seq $nj);do
                split_scps="${split_scps} ${splitdir}/feats.${n}.scp"
            done
            utils/split_scp.pl $featscp $split_scps || exit 1
        fi

        outemb=dump/${extract_set}/emb.${embname}
        ${train_cmd} JOB=1:${nj} log/speakerid_decode.${embname}.${extract_set}.JOB.log \
        speakerid_decode.py \
            --backend ${backend} \
            --ngpu ${ngpu} \
            --verbose ${verbose} \
            --out placeholder \
            --json placeholder \
            --model ${expdir}/results/${model} \
            --config ${decode_config} \
            --feat-scp ${splitdir}/feats.JOB.scp \
            --out-file ${outemb}.spltixJOB # if files with same names exist, files are overwritten from the beginning

        #cat ${outemb}.* | sort -k1 > ${outemb}
        #cat ${outemb}.spltix* | sort -k1 > ${outemb} # cat does NOT cause error for catting a to a large file. it might be the sort part so I added "-T" option to it as below to avoid space (I think memory) error
        cat ${outemb}.spltix* | sort -k1 -T tmp_sort/ > ${outemb}
        rm ${outemb}.spltix*
        # (TODO: Remove ${outemb}.*)
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

