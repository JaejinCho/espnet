#!/bin/bash

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
stage=5
stop_stage=100
ngpu=0       # number of gpu in training
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
model=model.loss.best # or give just a specific snapshot (without any path info. e.g. snapshot.ep.6)
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Speaker id related
spkloss_weight=0

# Speaker id backend related
expdir=""

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/60

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

## JJ - start
#if [ ${n_average} -gt 0 ]; then
#    model=model.last${n_average}.avg.best
#fi
path_500k_data=/export/b13/jcho/espnet_v3/tools/kaldi/egs/voxceleb/v2/data/voxceleb2_500k
path_500k_dump=dump/voxceleb2_500k
eval_set=voxceleb1_test
tag=_backendtrain_with_voxceleb2_500k

if [[ model.loss.best == ${model} ]]; then
    embname=`basename ${expdir}`
else
    embname=`basename ${expdir}`_${model}
fi

echo Embedding name: ${embname}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Get embeddings from 500k random utterances from voxceleb2 (no data aug). Assuming the all embeddings are already extracted from voxceleb_train_combined"
    mkdir -p ${path_500k_dump}
    echo "utts in this directory is for SV backend training" >> ${path_500k_dump}/Readme.txt

    path_uttlist=${path_500k_data}/uttlist
    path_ori_emb=dump/voxceleb_train_combined_trainNdev/emb.${embname}
    filter_scp.pl ${path_uttlist} ${path_ori_emb} > ${path_500k_dump}/emb.${embname}
    wc -l ${path_500k_dump}/emb.${embname}
    echo "The above result (# emb) should be 500k"
fi

# stage 8
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: backend-training"
  ${train_cmd_bc} ${path_500k_dump}/log/compute_emb_mean_${embname}.log \
      ivector-mean ark:${path_500k_dump}/emb.${embname} \
      ${path_500k_dump}/emb_mean.vec.${embname} || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd_bc ${path_500k_dump}/log/lda.${embname}.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean ark:${path_500k_dump}/emb.${embname} ark:- |" \
    ark:${path_500k_data}/utt2spk ${path_500k_dump}/emb_transform.mat.${embname} || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd_bc ${path_500k_dump}/log/plda.${embname}.log \
    ivector-compute-plda ark:${path_500k_data}/spk2utt \
    "ark:ivector-subtract-global-mean ark:${path_500k_dump}/emb.${embname} ark:- | transform-vec ${path_500k_dump}/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${path_500k_dump}/plda.${embname} || exit 1;
fi

# stage 9
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 7: Calculate score"
  # Get results using the out-of-domain PLDA model.
  $train_cmd_bc exp/scores/log/spkid_scoring_${embname}.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${path_500k_dump}/plda.${embname} - |" \
    "ark:ivector-subtract-global-mean ${path_500k_dump}/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec ${path_500k_dump}/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${path_500k_dump}/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec ${path_500k_dump}/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat ../asrtts+spkid/data/${eval_set}/trials | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_${eval_set}_embname${embname}${tag} || exit 1;
fi

# stage 10
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "stage 8: EER & MinDCF"
  qsub -cwd -o log/qsub.run.eer_mindcf.voxceleb1.${embname}.log -e log/qsub.run.eer_mindcf.voxceleb1.${embname}.log -l mem_free=200G,ram_free=200G run.eer_mindcf.voxceleb1.sh --eval_set ${eval_set} --embname ${embname} --tag ${tag}
fi
### JJ - end
