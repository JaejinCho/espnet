#!/bin/bash

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

train_set_ori=../tts+spkid_final/data/train_clean_460 # JJ: If train_clean_460_org is used (long duration utts are not filtered), CUDA memory error seems to happen
train_set=train_train
dev_set=train_dev
#eval_set=train_eval

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    if [ ! -d ${train_set_ori} ]; then
        echo "Generate ${train_set_ori} directory first" && exit 1
    fi

    cp_name=`echo ${train_set_ori} | awk -F'/' '{print $NF}'`
    if [ ! -d data/${cp_name} ]; then
        mkdir -p data && cp -r ${train_set_ori} data
    fi

    ### JJ add - start (TODO: Code to replace text with text.asr)
    for subset in ${train_set} ${dev_set};do
        if [ ! -d data/${subset}_Xformer ];then
            cp -r data/${subset} data/${subset}_Xformer
            cp /export/b18/jcho/espnet_asr_transformer_final/egs/librispeech/asr1/data/${subset}/text.asr data/${subset}_Xformer/text
        else
            echo Directory exist already: data/${subset}_Xformer && exit 0
        fi
    done
    ### JJ add _ end
fi


feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}_Xformer
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}_Xformer
#feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}_Xformer


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation (Assuming features are already generated. Only part to care about is to generate .json with new text files)"
    echo "Do NOTHING in this stage"
fi

dict=data/lang_1char/${train_set}_Xformer_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}_Xformer/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set}_Xformer ${dict} > ${feat_tr_dir}_Xformer/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set}_Xformer ${dict} > ${feat_dt_dir}_Xformer/data.json
    ## JJ: currently comments this line for eval_set since asr transcripts were not extracted for this
    #data2json.sh --feat ${feat_ev_dir}/feats.scp \
    #     data/${eval_set}_Xformer ${dict} > ${feat_ev_dir}_Xformer/data.json

    # json prep. for speaker id module
    for subset in ${train_set}_Xformer ${dev_set}_Xformer; do
        # update json
        local/update_json_general.sh --loc output --k spklab ${dumpdir}/${subset}/data.json data/train_clean_460/train_trainNdev.utt2spklab # utt2spklab generated same way as for utt2spkid from Nanxin's Resnet code
    done
fi
num_spk=`cat data/${train_set}_Xformer/spk2utt | cut -d'_' -f1 | sort -u | wc -l` # should be 1k for now
if [ ! $num_spk -eq 1000 ];then
    echo "num speaker: ${num_spk} is supposed to be 1k." && exit 1
fi

echo The number of speakers for spkid training: ${num_spk}

if [ -z ${tag} ]; then
    expname=${train_set}_Xformer_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_Xformer_${backend}_${tag}
fi

expname=${expname}_spkloss_weight${spkloss_weight}

expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}_Xformer/data.json
    dt_json=${feat_dt_dir}_Xformer/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train_speakerid.py \
           --model-module espnet.nets.pytorch_backend.e2e_tts_tacotron2_speakerid:Tacotron2 \
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
    for subset in ${dev_set}_Xformer; do
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
    for subset in ${dev_set}_Xformer; do
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
    for subset in ${dev_set}_Xformer ${train_set}_Xformer; do
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
