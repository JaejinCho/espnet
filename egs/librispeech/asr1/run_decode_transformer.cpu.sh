#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=6               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
libritts_dir=/export/b18/jcho/espnet3/egs/libritts/tts+spkid_final
libritts_train_set=train_train
libritts_train_dev=train_dev
# move data
if [ ${stage} -le 0 ]; then
    echo "stage 0: Move data dirs from different corpus"
for subset in ${libritts_train_set} ${libritts_train_dev};do
    cp -r ${libritts_dir}/data/${subset} data/
done
cp /export/b11/jcho/espnet/egs/wsj/asr1/steps/make_fbank_pitch_libritts.sh steps/
cp /export/b11/jcho/espnet/egs/wsj/asr1/conf/{fbank,pitch}_libritts.conf conf/ # for make_fbank_pitch_libritts.sh
fi

libritts_feat_tr_dir=${dumpdir}/${libritts_train_set}/delta${do_delta}; mkdir -p ${libritts_feat_tr_dir}
libritts_feat_dt_dir=${dumpdir}/${libritts_train_dev}/delta${do_delta}; mkdir -p ${libritts_feat_dt_dir}

# generate dump (fine to use human transcripts here but ignore them later)
## feats.scp
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for subset in ${libritts_train_set} ${libritts_train_dev};do
        steps/make_fbank_pitch_libritts.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${subset} exp/make_fbank/${subset} ${fbankdir}
        utils/fix_data_dir.sh data/${subset}
    done

    # compute global CMVN (already done)
    #compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${libritts_feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${libritts_train_set}/delta${do_delta}/storage \
        ${libritts_feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${libritts_feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${libritts_train_dev}/delta${do_delta}/storage \
        ${libritts_feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 80 --do_delta ${do_delta} \
        data/${libritts_train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/libritts_train ${libritts_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${libritts_train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/libritts_dev ${libritts_feat_dt_dir}
fi
## data.json
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Make json (data.json) files"
    # make json labels
    data2json.sh --feat ${libritts_feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${libritts_train_set} ${dict} > ${libritts_feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${libritts_feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${libritts_train_dev} ${dict} > ${libritts_feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

train_config=conf/tuning/train_pytorch_transformer_large_ngpu4.yam # might not be used for decoding
decode_config=conf/tuning/decode_pytorch_transformer_large.yaml
expdir=exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
recog_model=model.val5.avg.best
lmexpdir=exp/irielm.ep11.last5.avg
lang_model=rnnlm.model.best
lmtag=pretrained
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
