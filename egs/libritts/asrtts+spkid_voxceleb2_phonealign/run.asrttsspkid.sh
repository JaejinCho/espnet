#!/bin/bash

# JJ: Edited from /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb_phonealign/run.asrttsspkid.sh
# JJ: Edited from /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxcelebNaug_kaldiasr_phonealign_rf3/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addnceloss.sh
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
cuda_memsize=5 # this means 5. The other option is 20

# feature extraction related (set by following /export/b18/jcho/espnet3/egs/libritts/tts_featext/run.FE.diffconfig.voxceleb1.sh)
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points (32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=160   # number of shift points (10ms)
win_length="" # window length

# config files
train_config=conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching.yaml
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
train_set=voxceleb2_train
dev_set=voxceleb2_dev
eval_set=voxceleb1_test

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    if [ ! -d data ]; then
        mkdir -p data
    fi
    # 0. Copy needed data
    if [ ! -d data/voxceleb2 ]; then
        cp -r /export/b13/jcho/espnet_v3/tools/kaldi/egs/voxceleb/v2/data/voxceleb2 data/
        if [ ! -f data/voxceleb2/feats.scp ]; then
            filter_scp.pl data/voxceleb2/uttlist ${feat_dir}/data/voxceleb_train/feats.scp > data/voxceleb2/feats.scp
        fi
    fi
    if [ ! -d data/${eval_set} ]; then
        cp -r ${feat_dir}/data/voxceleb1_test data/
    fi
    # 1. Prepare text and remove ${eval_set} portion from the training data
    ## Get ori text file (voxcelelb2_all: voxceleb2_{train,test})
    cat /export/b11/jcho/kaldi_20190316/egs/librispeech/s5/exp/chain_cleaned/tdnn_1d_sp/decode_voxceleb2_all_frame_subsampling_factor_3/phone.perframe.sym.txt > data/voxceleb2/text
    #mv data/voxceleb2/text data/voxceleb2/.text && awk '{if(!(NF==1))print $0}' data/voxceleb2/.text > data/voxceleb2/text # JJ: This one does nothing in the kaldiali case
    utils/fix_data_dir.sh data/voxceleb2

    # 2. Divide data/voxceleb2 into data/voxceleb2_{train,dev}
    if [ ! -f ./data/voxceleb2/${train_set}.uttlist ] && [ ! -f ./data/voxceleb2/${dev_set}.uttlist ] && [ ! -f ./data/voxceleb2/utt2spklab ] ; then
        separate_trainNdev_uttlists.py voxceleb2 ${train_set} ${dev_set} 0.04
    fi
    ## Extract subsets for train_train and train_dev
    for subset in ${train_set} ${dev_set}; do
        if [ ! -d data/${subset} ]; then
            subset_data_dir.sh --utt-list data/voxceleb2/${subset}.uttlist data/voxceleb2 data/${subset}
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
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark # I can use whole voxceleb2 data for this but this won't make a big difference with this large amount of data

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
    cp /export/b11/jcho/kaldi_20190316/egs/librispeech/s5/exp/chain_cleaned/tdnn_1d_sp/phones.txt ${dict}
    sed -e '1 s:<eps> 0:<unk> 0:' ${dict} | awk -F' ' '{$2=$2+1;print}' OFS=' ' > .tmpdict && mv .tmpdict ${dict}
    wc -l ${dict}

    # make json labels
    data2json_kaldiali.sh --kaldiali true --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json_kaldiali.sh --kaldiali true --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    ## JJ: currently comments out lines below
    #data2json_kaldiali.sh --kaldiali true --feat ${feat_ev_dir}/feats.scp \
    #     data/${eval_set} ${dict} > ${feat_ev_dir}/data.json

    # json prep. for speaker id module
    for subset in ${train_set} ${dev_set}; do
        # update json
        local/update_json_general.sh --loc output --k spklab ${dumpdir}/${subset}/data.json ./data/voxceleb2/utt2spklab
    done
fi

num_spk=`cat data/${train_set}/spk2utt | wc -l`
if [ ! $num_spk -eq 6114 ];then
    echo "num speaker: ${num_spk} is supposed to be 6114." && exit 1
fi
echo The number of speakers for spkid training: ${num_spk} # should be 6114 for voxceleb2

if [ -z ${tag} ]; then
    expname=${train_set}_$(basename ${train_config%.*})
else
    expname=${train_set}_$(basename ${train_config%.*})_${tag}
fi

expname=${expname}_spkloss_weight${spkloss_weight}_fullyunsync

expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    if [ ${cuda_memsize} == 5 ]; then
        cuda_cmd_train=${cuda_cmd_mem5G}
    elif [ ${cuda_memsize} == 10 ];then
        cuda_cmd_train=${cuda_cmd_mem10G}
    elif [ ${cuda_memsize} == 20 ];then
        cuda_cmd_train=${cuda_cmd_mem20G}
    else
        cuda_cmd_train=${cuda_cmd}
    fi
    echo "cuda_cmd_train: ${cuda_cmd_train}"
    ${cuda_cmd_train} --gpu ${ngpu} ${expdir}/train.log \
        tts_train_speakerid.py \
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
