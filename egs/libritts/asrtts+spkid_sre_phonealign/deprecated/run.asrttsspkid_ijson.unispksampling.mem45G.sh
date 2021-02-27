#!/bin/bash

# JJ: Edited from run.asrttsspkid_ijson.mem30G.sh 
# JJ: Edited from run.asrttsspkid.mem20G.sh
# JJ: Edited from /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/run.asrttsspkid.sh
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
cuda_memsize=45

# feature extraction related (set by following /export/b18/jcho/espnet3/egs/libritts/tts_featext/run.FE.diffconfig.voxceleb1.sh)
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points (32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=160   # number of shift points (10ms)
win_length="" # window length

# config files
train_config=conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling.yaml
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

corpora_list="babel_amharic babel_assamese babel_bengali babel_cantonese babel_cebuano babel_dholuo babel_georgian babel_guarani babel_haitian babel_igbo babel_javanese babel_kazakh babel_kurmanji babel_lao babel_lithuanian babel_mongolian babel_pashto babel_swahili babel_tagalog babel_tamil babel_telugu babel_tokpisin babel_turkish babel_vietnamese babel_zulu fisher_spa sre16_eval_tr60_tgl sre16_eval_tr60_yue sre16_train_dev_ceb sre18_cmn2_train_lab sre_tel swbd voxcelebcat_tel sre16_train_dev_cmn"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    if [ ! -d data ]; then
        mkdir -p data
    fi
    # 0. Copy needed data (TODO)
    ## 0.1 data dir w/ text (each corpus has "feats.scp  filetype  segments  spk2utt  utt2num_frames  utt2spk  wav.scp")
    for cname in ${corpora_list}; do
        # cp data dir w/o text
        cp -r ${feat_dir}/data/${cname} data/
        # cp text
        cp ${trans_dir}/decode_${cname}_frame_subsampling_factor_3/phone.perframe.sym.txt data/${cname}/text
    done

    # 1. Divide data into train and dev for NN training (Do this separately for lab and nolab corpora) (TODO)
    ## 1-1 lab (corpora except babel)
    start_spkidx=0
    for cname in `echo $corpora_list | tr ' ' '\n' | grep -v babel | tr '\n' ' '`;do
        if [ ! -f ./data/${cname}/${cname}_train.uttlist ] && [ ! -f ./data/${cname}/${cname}_dev.uttlist ] && [ ! -f ./data/${cname}/utt2spklab ] ; then
            ## continue index for utt2spklab files to distinguish speaker indicies over multiple corpora
            python separate_trainNdev_uttlists_startspkidx.py ${cname} ${cname}_train ${cname}_dev 0.04 ${start_spkidx} 2>&1 | tee log/separate_trainNdev_uttlists_startspkidx.${cname}.start_spkidx${start_spkidx}.log
            start_spkidx=$(grep "start_spkidx" log/separate_trainNdev_uttlists_startspkidx.${cname}.start_spkidx${start_spkidx}.log | awk -F':' '{print $NF}')
        fi
    done
    ## 1-2 nolab (babel) - I think it is NOT needed to divide since there is no ground-truth spk id labels avail so cannot include this for validation
    ## Extract subsets for *_train and *_dev
    for cname in `echo $corpora_list | tr ' ' '\n' | grep -v babel | tr '\n' ' '`;do
        for subset in ${cname}_train ${cname}_dev; do
            if [ ! -d data/${subset} ]; then
                subset_data_dir.sh --utt-list data/${cname}/${subset}.uttlist data/${cname} data/${subset}
            fi
        done
    done
fi


for cname in `echo $corpora_list`;do
    if [[ "$cname" == *"babel"* ]]; then
        mkdir -p ${dumpdir}/${cname}_nolab
    else
        mkdir -p ${dumpdir}/${cname}_train
        mkdir -p ${dumpdir}/${cname}_dev
    fi
done


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation (dump using apply-cmvn-sliding)"
    pids=() # initialize pids
    for cname in `echo $corpora_list`;do
    (
        if [[ "$cname" == *"babel"* ]]; then
            echo "NOLAB: $cname"
            dump_cmvnsliding.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
                data/${cname}/feats.scp exp/dump_feats/${cname}_nolab ${dumpdir}/${cname}_nolab
        else
            echo "LAB: $cname"
            dump_cmvnsliding.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
                data/${cname}_train/feats.scp exp/dump_feats/${cname}_train ${dumpdir}/${cname}_train
            dump_cmvnsliding.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
                data/${cname}_dev/feats.scp exp/dump_feats/${cname}_dev ${dumpdir}/${cname}_dev
        # (TODO): Prepare eval_set
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

### CHECKING - start ###
dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    cp ${trans_dir}/phones.txt ${dict}
    sed -e '1 s:<eps> 0:<unk> 0:' ${dict} | awk -F' ' '{$2=$2+1;print}' OFS=' ' > .tmpdict && mv .tmpdict ${dict}
    wc -l ${dict}

    # make json labels
    pids=()
    for cname in `echo $corpora_list`;do
    (
        if [[ "$cname" == *"babel"* ]]; then
            echo "NOLAB: $cname"
            qsub -wd /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sre_phonealign -o log/data2json_kaldiali.qsub.${cname}_nolab.log -e log/data2json_kaldiali.qsub.${cname}_nolab.log data2json_kaldiali.qsub.sh ${dumpdir} ${cname}_nolab ${dict}
            #data2json_kaldiali.sh --kaldiali true --feat ${dumpdir}/${cname}_nolab/feats.scp \
            #    data/${cname} ${dict} > ${dumpdir}/${cname}_nolab/data.json
        else
            echo "LAB: $cname"
            qsub -wd /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sre_phonealign -o log/data2json_kaldiali.qsub.${cname}_train.log -e log/data2json_kaldiali.qsub.${cname}_train.log data2json_kaldiali.qsub.sh ${dumpdir} ${cname}_train ${dict}
            qsub -wd /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sre_phonealign -o log/data2json_kaldiali.qsub.${cname}_dev.log -e log/data2json_kaldiali.qsub.${cname}_dev.log data2json_kaldiali.qsub.sh ${dumpdir} ${cname}_dev ${dict}
            #data2json_kaldiali.sh --kaldiali true --feat ${dumpdir}/${cname}_train/feats.scp \
            #    data/${cname}_train ${dict} > ${dumpdir}/${cname}_train/data.json
            #data2json_kaldiali.sh --kaldiali true --feat ${dumpdir}/${cname}_dev/feats.scp \
            #    data/${cname}_dev ${dict} > ${dumpdir}/${cname}_dev/data.json
        fi
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."

    # json prep. for speaker id module
    pids=()
    for cname in `echo $corpora_list`;do
    (
        if [[ "$cname" == *"babel"* ]]; then
            echo "NOLAB: $cname"
            awk '{print $1, -1}' ./data/${cname}/utt2spk > ./data/${cname}/utt2spklab
            qsub -wd /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sre_phonealign -o log/update_json_general.qsub.${cname}_nolab.log -e log/update_json_general.qsub.${cname}_nolab.log local/update_json_general.sh --loc output --k spklab ${dumpdir}/${cname}_nolab/data.json ./data/${cname}/utt2spklab
        else
            echo "LAB: $cname"
            qsub -wd /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sre_phonealign -o log/update_json_general.qsub.${cname}_train.log -e log/update_json_general.qsub.${cname}_train.log local/update_json_general.sh --loc output --k spklab ${dumpdir}/${cname}_train/data.json ./data/${cname}/utt2spklab
            qsub -wd /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sre_phonealign -o log/update_json_general.qsub.${cname}_dev.log -e log/update_json_general.qsub.${cname}_dev.log local/update_json_general.sh --loc output --k spklab ${dumpdir}/${cname}_dev/data.json ./data/${cname}/utt2spklab
            #local/update_json_general.sh --loc output --k spklab ${dumpdir}/${cname}_train/data.json ./data/${cname}/utt2spklab
            #local/update_json_general.sh --loc output --k spklab ${dumpdir}/${cname}_dev/data.json ./data/${cname}/utt2spklab
        fi
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
### CHECKING - end ###

num_spk=0
for corpora in `echo $corpora_list`; do
    if [[ "$corpora" != *"babel"* ]]; then
        temp_num=$(wc -l data/${corpora}/spk2utt | awk '{print $1}')
        num_spk=$((num_spk+temp_num))
    fi
done
echo The number of speakers for spkid training: ${num_spk} # 15072 for sre20 recipe

if [ -z ${tag} ]; then
    expname=${train_set}_$(basename ${train_config%.*})
else
    expname=${train_set}_$(basename ${train_config%.*})_${tag}
fi

expname=${expname}_spkloss_weight${spkloss_weight}_fullyunsync_mem45G_ijson

expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    # build tr_json* and dt_json
    tr_json_nolab=""
    tr_json=""
    dt_json=""
    for cname in `echo $corpora_list`;do
        if [[ "$cname" == *"babel"* ]]; then
            tr_json_nolab+="${dumpdir}/${cname}_nolab/data.json "
        else
            echo "LAB: $cname"
            tr_json+="${dumpdir}/${cname}_train/data.json "
            dt_json+="${dumpdir}/${cname}_dev/data.json "
        fi
    done

    # actual run
    if [ ${cuda_memsize} == 5 ]; then
        cuda_cmd_train=${cuda_cmd_mem5G}
    elif [ ${cuda_memsize} == 10 ];then
        cuda_cmd_train=${cuda_cmd_mem10G}
    elif [ ${cuda_memsize} == 45 ];then
        cuda_cmd_train=${cuda_cmd_mem45G}
    else
        cuda_cmd_train=${cuda_cmd}
    fi
    echo "cuda_cmd_train: ${cuda_cmd_train}"
    ${cuda_cmd_train} --gpu ${ngpu} ${expdir}/train.log \
        tts_train_speakerid_semi_multicorpora_ijson.py \
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
           --train-json-nolab "${tr_json_nolab}" \
           --train-json "${tr_json}" \
           --valid-json "${dt_json}" \
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
