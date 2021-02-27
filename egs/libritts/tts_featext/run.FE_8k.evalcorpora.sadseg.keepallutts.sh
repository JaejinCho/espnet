#!/bin/bash

# Edited from run.FE_8k.sre19_eval_cmn2.sadseg.keepallutts.sh to expand it to
# all eval corpora
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
fs=8000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_fft=256    # number of fft points (~32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=80   # number of shift points (~10ms)
win_length="" # window length
n_mels=64     # number of mel basis

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

kaldi_datadir=/export/b16/jcho/hyperion_trials/egs/sre20-cts/v1/data
data_list="sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation. Copy parts of data dirs & Get segments using SAD"
    pids=() # initialize pids
    # decode in parallel with cpus
    for dataname in ${data_list}; do
    (
        if [ ! -d data/${dataname}_NOdiscard ]; then
            echo "Copying ${kaldi_datadir}/${dataname}/{wav.scp,utt2spk,spk2utt} to data/${dataname}_NOdiscard/"
            mkdir -p data/${dataname}_NOdiscard
            cp ${kaldi_datadir}/${dataname}/{wav.scp,utt2spk,spk2utt} data/${dataname}_NOdiscard/
        fi
        # SAD segments
        if [ ! -s data/${dataname}_NOdiscard/segments ]; then
            echo "Getting SAD segments and write to data/${dataname}_NOdiscard/segments. Also generating new utt2spk and spk2utt accordingly"
            conf_dir=/export/b11/jcho/kaldi_20190316/egs/librispeech/s5
            temp_nj=$(wc -l data/${dataname}_NOdiscard/spk2utt | awk '{print $1}')
            if [[ ${temp_nj} -le ${nj} ]]; then
                steps/segmentation/detect_speech_activity_keepallutts.sh --nj ${temp_nj} --extra-left-context 79 --extra-right-context 21 --extra-left-context-initial 0 --extra-right-context-final 0 --frames-per-chunk 150 --mfcc-config ${conf_dir}/SAD_related/vimal_SAD/conf/mfcc_hires.conf data/${dataname}_NOdiscard ${conf_dir}/SAD_related/vimal_SAD/exp/segmentation_1a/tdnn_stats_asr_sad_1a ${conf_dir}/SAD_related/vimal_SAD/sre20_train_prep_${dataname}_NOdiscard_mfcc ${conf_dir}/SAD_related/vimal_SAD/sre20_train_prep_${dataname}_NOdiscard_work data/${dataname}_NOdiscard
            else
                steps/segmentation/detect_speech_activity_keepallutts.sh --nj ${nj} --extra-left-context 79 --extra-right-context 21 --extra-left-context-initial 0 --extra-right-context-final 0 --frames-per-chunk 150 --mfcc-config ${conf_dir}/SAD_related/vimal_SAD/conf/mfcc_hires.conf data/${dataname}_NOdiscard ${conf_dir}/SAD_related/vimal_SAD/exp/segmentation_1a/tdnn_stats_asr_sad_1a ${conf_dir}/SAD_related/vimal_SAD/sre20_train_prep_${dataname}_NOdiscard_mfcc ${conf_dir}/SAD_related/vimal_SAD/sre20_train_prep_${dataname}_NOdiscard_work data/${dataname}_NOdiscard
            fi
            # This far, there are only "spk2utt  utt2spk  wav.scp" in the dir,
            # data/${dataname}_NOdiscard. Below will generate segments and the new {wav.scp,utt2spk,spk2utt} there
            mv data/${dataname}_NOdiscard/wav.scp data/${dataname}_NOdiscard/.wav.scp && cp data/${dataname}_NOdiscard_seg/wav.scp data/${dataname}_NOdiscard/wav.scp # (newly added but not run for others)
            python local/seg2newseg.py data/${dataname}_NOdiscard_seg/segments data/${dataname}_NOdiscard/segments 0.5 4 0 # NO discard any segments
            python local/utt2spk2newutt2spk.py data/${dataname}_NOdiscard/segments data/${dataname}_NOdiscard/utt2spk data/${dataname}_NOdiscard/new_utt2spk # I think in the 2nd column of segments files are [spkid]-[uttid] but I just created a python script.
            utt2spk_to_spk2utt.pl data/${dataname}_NOdiscard/new_utt2spk > data/${dataname}_NOdiscard/new_spk2utt
            mv data/${dataname}_NOdiscard/utt2spk data/${dataname}_NOdiscard/.utt2spk && mv data/${dataname}_NOdiscard/new_utt2spk data/${dataname}_NOdiscard/utt2spk
            mv data/${dataname}_NOdiscard/spk2utt data/${dataname}_NOdiscard/.spk2utt && mv data/${dataname}_NOdiscard/new_spk2utt data/${dataname}_NOdiscard/spk2utt
        fi
        fix_data_dir.sh data/${dataname}_NOdiscard # detect_speech_activity*.sh also include this step
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation (used espnet version NOT kaldi to be consistent with wavform synthesis later using griffi_lim"

    fbankdir=fbank
    set_nj=${nj}
    for dataname in ${data_list}; do
    (
        if [ ! -s data/${dataname}_NOdiscard/feats.scp ]; then
            temp_nj=$(wc -l data/${dataname}_NOdiscard/spk2utt | awk '{print $1}')
            if [[ ${temp_nj} -le ${set_nj} ]]; then
                nj=${temp_nj}
            else
                nj=${set_nj}
            fi
            make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                data/${dataname}_NOdiscard \
                exp/make_fbank/${dataname}_NOdiscard \
                ${fbankdir}
            fix_data_dir.sh data/${dataname}_NOdiscard
        else
            echo "WARNING: There exist feats.scp already under data/${dataname}_NOdiscard so skipping the feature extraction"
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
