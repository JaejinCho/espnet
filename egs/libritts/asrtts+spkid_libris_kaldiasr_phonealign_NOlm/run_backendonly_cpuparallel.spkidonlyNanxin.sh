#!/bin/bash

# JJ: change only model (or give different parameters to model) for different embedding backend training and evaluation
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
#model=model.loss.best
model=11_98.h5
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Speaker id related
spkloss_weight=0

# Speaker id backend related
#expdir=exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_speakeridonly_ttslossw0_spklossw1
expdir=exp/spkidonly_nanxincode


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

## JJ - start
#if [ ${n_average} -gt 0 ]; then
#    model=model.last${n_average}.avg.best
#fi
train_set=train_train
dev_set=train_dev
eval_set=train_eval

if [ ! -f data/${eval_set}/trials ]; then
  python compose_trials.py data/${eval_set}/spk2gender data/${eval_set}/utt2spk # the output trials will be written to data/train_eval
fi

if [[ model.loss.best == ${model} ]]; then
    embname=`basename ${expdir}`
else
    embname=`basename ${expdir}`_${model}
fi

echo Embedding name: ${embname}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Copying decoded embedding"
    emb_ori_dir=/export/b18/jcho/libritts_spkidonly/ivs

    for subset in ${train_set} ${dev_set} ${eval_set};do
        cp ${emb_ori_dir}/embedding.${subset}.from${model}.ark dump/${subset}/emb.${embname}
    done
fi

# stage 8
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: backend-training"
  if [ ! -d dump/train_trainNdev ]; then
    mkdir -p dump/train_trainNdev
  fi
  cat dump/train_{train,dev}/emb.${embname} > dump/train_trainNdev/emb.${embname}

  # to combine utt2spk and spk2utt
  if [ ! -d data/train_trainNdev ]; then
    combine_data.sh data/train_trainNdev data/train_train data/train_dev
  fi

  if [ ! -f data/train_trainNdev/utt2spk_actual ]; then
      cat data/train_trainNdev/utt2spk | rev | cut -d'_' -f2- | rev > data/train_trainNdev/utt2spk_actual
      utt2spk_to_spk2utt.pl data/train_trainNdev/utt2spk_actual > data/train_trainNdev/spk2utt_actual
  fi

  $train_cmd dump/train_trainNdev/log/compute_emb_mean_${embname}.log \
      ivector-mean ark:dump/train_trainNdev/emb.${embname} \
      dump/train_trainNdev/emb_mean.vec.${embname} || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  # JJ (TODO): utt2spk should be changed to make spk actual spk. (Also think
  # about building trials wihtout grouping same speakers in enrollment (current
  # way) since this lda actually groups speakers and calculate this
  # transformation)
  lda_dim=150
  $train_cmd dump/train_trainNdev/log/lda.${embname}.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean ark:dump/train_trainNdev/emb.${embname} ark:- |" \
    ark:data/train_trainNdev/utt2spk_actual dump/train_trainNdev/emb_transform.mat.${embname} || exit 1;

  # Train an out-of-domain PLDA model. JJ (TODO): fix spk2utt same as the above
  # for ivector-compute-lda (also think about current way of composing trials)
  $train_cmd dump/train_trainNdev/log/plda.${embname}.log \
    ivector-compute-plda ark:data/train_trainNdev/spk2utt_actual \
    "ark:ivector-subtract-global-mean ark:dump/train_trainNdev/emb.${embname} ark:- | transform-vec dump/train_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    dump/train_trainNdev/plda.${embname} || exit 1;
fi
# stage 9
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 7: Calculate score"
  # Get results using the out-of-domain PLDA model.
  $train_cmd exp/scores/log/spkid_scoring_${embname}.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 dump/train_trainNdev/plda.${embname} - |" \
    "ark:ivector-subtract-global-mean dump/train_trainNdev/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec dump/train_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean dump/train_trainNdev/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec dump/train_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat data/${eval_set}/trials | cut -d\  --fields=1,2 |" exp/scores_libritts_${eval_set}_embname${embname} || exit 1;
fi
# stage 10
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "stage 8: EER & MinDCF"
  #eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  #mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  #mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  #eer=`compute-eer <(local/pre data/${eval_set}/trials exp/scores_libritts_${eval_set}_embname${embname}) 2> /dev/null` # JJ: error raises. I think due to # trials are too large
  echo "Computing EER..."
  paste data/${eval_set}/trials exp/scores_libritts_${eval_set}_embname${embname} | awk '{print $6, $3}' | compute-eer -
  #eer=$(paste data/${eval_set}/trials exp/scores_libritts_${eval_set}_embname${embname} | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  #mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_libritts_${eval_set}_embname${embname} data/${eval_set}/trials 2> /dev/null`
  #mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_libritts_${eval_set}_embname${embname} data/${eval_set}/trials 2> /dev/null`
  #echo "EER: $eer%"
  #echo "minDCF(p-target=0.01): $mindcf1"
  #echo "minDCF(p-target=0.001): $mindcf2"
fi
### JJ - end
