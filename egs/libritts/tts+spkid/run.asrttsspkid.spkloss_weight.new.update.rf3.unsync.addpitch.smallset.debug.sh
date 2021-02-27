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
gpu_ix=''
nj=64        # numebr of parallel jobs
dumpdir=dump_smallset # directory to dump full features
verbose=0    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=24000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# config files
train_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Speaker id related
spkloss_weight=0

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

train_set_ori=train_clean_460
train_set=train_clean_460_spktrain_0.9
dev_set=train_clean_460_spkcv_0.1
eval_set=test_clean

# This script assumes data dirs and dump dirs for train_set_ori, train_set, dev_set, eval_set were generated already

feat_tr_dir=${dumpdir}/${train_set}
feat_dt_dir=${dumpdir}/${dev_set}
feat_ev_dir=${dumpdir}/${eval_set}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Pitch Generation"
    # pitch in data dir
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
    # copy dump files for pitch-added version & add pitch features to data.json
    for x in ${train_set} ${dev_set} ${eval_set};do
        cp -r ${dumpdir}/${x} ${dumpdir}/${x}_addpitch
        cp data/${x}/pitch.scp data/${x}/.pitch.txt
        local/update_json_general.sh --loc input --k pitch ${dumpdir}/${x}_addpitch/data.json data/${x}/.pitch.txt # featdir: some dump dir
    done
fi

num_spk=`awk 'BEGIN {s=0;} {if($2>s)s+=1} END{print s+1}' data/${train_set_ori}/utt2spklab`
echo Number of speaker for spkid training: ${num_spk}

if [ -z ${tag} ]; then
    expname=${train_set}_addpitch_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_addpitch_${backend}_${tag}
fi
expname=${expname}_smallset_debug
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}_addpitch/data.json
    dt_json=${feat_dt_dir}_addpitch/data.json
    echo "tr_json: ${tr_json}"
    echo "dt_json: ${dt_json}"
    #${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=${gpu_ix} tts_train_speakerid_addpitch.py \
           --model-module espnet.nets.pytorch_backend.e2e_tts_tacotron2_speakerid_update_unsync_addpitch:Tacotron2 \
           --num-spk ${num_spk} \
           --train-spkid-extractor True \
           --train-spk-embed-dim 400 \
           --spkloss-weight ${spkloss_weight} \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for subset in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${subset} ] && mkdir -p ${outdir}/${subset}
        cp ${dumpdir}/${subset}/data.json ${outdir}/${subset}
        splitjson.py --parts ${nj} ${outdir}/${subset}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${subset}/log/decode.JOB.log \
            tts_decode_speakerid.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${subset}/feats.JOB \
                --json ${outdir}/${subset}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --spkid-onthefly True \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${subset}/feats.$n.scp" || exit 1;
        done > ${outdir}/${subset}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis"
    pids=() # initialize pids
    for subset in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${subset} ] && mkdir -p ${outdir}_denorm/${subset}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${subset}/feats.scp \
            ark,scp:${outdir}_denorm/${subset}/feats.ark,${outdir}_denorm/${subset}/feats.scp
        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ${outdir}_denorm/${subset} \
            ${outdir}_denorm/${subset}/log \
            ${outdir}_denorm/${subset}/wav
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
