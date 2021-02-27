#!/bin/bash

# JJ: Edited from run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.sh
# JJ: Edited from run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.sh
# JJ: Edited from run.asrttsspkid.spkloss_weight.new.update.rf3.sh
# JJ: Copied and edited from run.asrttsspkid.spkloss_weight.new.update.rf1.sh

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# train set
train_set=voxceleb2_800spk # e.g) data/voxceleb2_{2400,1600,800}spk

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

# feature extraction related (set by following /export/b18/jcho/espnet3/egs/libritts/tts_featext/run.FE.diffconfig.voxceleb1.sh)
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points (32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=160   # number of shift points (10ms)
win_length="" # window length

# config files
train_config=conf/train_pytorch_tacotron2+spkemb_noatt_rf3.yaml
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

feat_dir=/export/b18/jcho/espnet3/egs/libritts/tts_featext
dev_set=voxceleb1_train_filtered_dev
eval_set=voxceleb1_test

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation assuming dev_set (and eval_set) are ready beforehand in other dir"
    mkdir -p data
    ## train_set
    # data dir
    if [ ! -d data/${train_set} ]; then
        cp -r ${feat_dir}/data/${train_set} data/${train_set}
        ls -A data/${train_set}
    fi
    # text file
    if [ ! -f data/${train_set}/text ]; then
        oritext=/export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxcelebNaug_kaldiasr_phonealign_rf3/data/voxceleb_train/text
        filter_scp.pl <(awk '{print $1}' data/${train_set}/utt2spk) ${oritext} > data/${train_set}/text
    fi
    echo "Before fix_data_dir.sh"
    wc -l data/${train_set}/*
    fix_data_dir.sh data/${train_set}
    echo "After fix_data_dir.sh"
    wc -l data/${train_set}/*
    ## dev_set and eval_set
    oridir=/export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3
    for subset in ${dev_set} ${eval_set}; do
        if [ ! -d data/${subset} ]; then
            cp -r ${oridir}/data/${subset} data/${subset}
        fi
    done
fi


feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    # JJ add - start
    cp /export/b11/jcho/kaldi_20190316/egs/librispeech/s5/exp/chain_cleaned/tdnn_1d_sp/phones.txt ${dict}
    sed -e '1 s:<eps> 0:<unk> 0:' ${dict} | awk -F' ' '{$2=$2+1;print}' OFS=' ' > .tmpdict && mv .tmpdict ${dict}
    # JJ add - end
    wc -l ${dict}

    # make json labels
    data2json_kaldiali.sh --kaldiali true --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json_kaldiali.sh --kaldiali true --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    ## JJ: currently comments this line
    #data2json_kaldiali.sh --kaldiali true --feat ${feat_ev_dir}/feats.scp \
    #     data/${eval_set} ${dict} > ${feat_ev_dir}/data.json

    # json prep. for speaker id module
    ## train_set
    ### fake utt2spklab (******* true utt2spklab is not need since it is for ttsonly setup)
    if [ ! -f data/${train_set}/utt2spklab ]; then
        #awk -F' ' '{print $1, 0}' data/${train_set}/utt2spk > data/${train_set}/utt2spklab # JJ: This seems to cause some error in evaluation at each epoch's end since we have unseen labels in dev_set
        awk 'BEGIN{s=0;}{if(s<1210){print $1,s;s+=1;}else{if(s==1210){print $1,s;s=0;}}}' data/${train_set}/utt2spk > data/${train_set}/utt2spklab
    fi
    local/update_json_general.sh --loc output --k spklab ${dumpdir}/${train_set}/data.json data/${train_set}/utt2spklab
    ## dev_set
    for subset in ${dev_set}; do
        # update json
        local/update_json_general.sh --loc output --k spklab ${dumpdir}/${subset}/data.json /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/data/voxceleb1_train_filtered_librispeechXformerASR/voxceleb1_train_filtered.utt2spklab # utt2spklab generated same way as for utt2spkid from Nanxin's Resnet code
    done
fi

num_spk=`cat data/${train_set}/spk2utt | wc -l`
if [ ! $num_spk -eq 800 ];then
    echo "num speaker: ${num_spk} is supposed to be 800." && exit 1
fi
echo The number of speakers for spkid training: ${num_spk} # should be 800
echo However, the num_spk given to tts_train_speakerid.py is 1211 to make the training work with utt2spklab # utt2spklab has label value from 0 to 1210
num_spk_utt2spklab=1211

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi

expname=${expname}_spkloss_weight${spkloss_weight}_fullyunsync

expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd_mem5G} --gpu ${ngpu} ${expdir}/train.log \
        tts_train_speakerid.py \
           --model-module espnet.nets.pytorch_backend.e2e_tts_tacotron2_speakerid_update_fullyunsync:Tacotron2 \
           --num-spk ${num_spk_utt2spklab} \
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
    #for subset in ${dev_set} ${eval_set}; do
    for subset in ${dev_set}; do
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
    #for subset in ${dev_set} ${eval_set}; do
    for subset in ${dev_set}; do
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

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: classification validation in speaker ID"
    for subset in ${dev_set} ${train_set}; do
        echo "Current Subset: ${subset}"
        ${cuda_cmd} --gpu ${ngpu} ${expdir}/spkid_clss_eval.${subset}.log \
        speakerid_eval_classification.py \
            --backend ${backend} \
            --ngpu ${ngpu} \
            --verbose ${verbose} \
            --json ${dumpdir}/${subset}/data.json \
            --model ${expdir}/results/${model} \
            --batch-size 256
    done
fi
