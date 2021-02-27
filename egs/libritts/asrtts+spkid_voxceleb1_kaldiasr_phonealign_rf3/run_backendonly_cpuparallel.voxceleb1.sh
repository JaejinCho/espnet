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
lda_dim=150

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
train_set=voxceleb1_train_filtered_train
dev_set=voxceleb1_train_filtered_dev
eval_set=voxceleb1_test

if [ ! -f data/${eval_set}/trials ]; then
    echo "Copying trials from ../asrtts+spkid/data/${eval_set}/trials"
    cp ../asrtts+spkid/data/${eval_set}/trials data/${eval_set}/
fi

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
    for extract_set in ${train_set} ${dev_set} ${eval_set}; do
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
        cat ${outemb}.spltix* | sort -k1 > ${outemb}
        rm ${outemb}.spltix*
        # (TODO: Remove ${outemb}.*)
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

# stage 8
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: backend-training"
  if [ ! -d dump/voxceleb1_train_filtered_trainNdev ]; then
    mkdir -p dump/voxceleb1_train_filtered_trainNdev
  fi
  if [ ! -f dump/voxceleb1_train_filtered_trainNdev/emb.${embname} ]; then
  cat dump/voxceleb1_train_filtered_{train,dev}/emb.${embname} > dump/voxceleb1_train_filtered_trainNdev/emb.${embname}
  rm dump/voxceleb1_train_filtered_{train,dev}/emb.${embname}
  fi

  # to combine utt2spk and spk2utt
  if [ ! -d data/voxceleb1_train_filtered_trainNdev ]; then
    combine_data.sh data/voxceleb1_train_filtered_trainNdev data/voxceleb1_train_filtered_train data/voxceleb1_train_filtered_dev
  fi


  $train_cmd dump/voxceleb1_train_filtered_trainNdev/log/compute_emb_mean_${embname}.log \
      ivector-mean ark:dump/voxceleb1_train_filtered_trainNdev/emb.${embname} \
      dump/voxceleb1_train_filtered_trainNdev/emb_mean.vec.${embname} || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=${lda_dim}
  $train_cmd dump/voxceleb1_train_filtered_trainNdev/log/lda.${embname}.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean ark:dump/voxceleb1_train_filtered_trainNdev/emb.${embname} ark:- |" \
    ark:data/voxceleb1_train_filtered_trainNdev/utt2spk dump/voxceleb1_train_filtered_trainNdev/emb_transform.mat.${embname} || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd dump/voxceleb1_train_filtered_trainNdev/log/plda.${embname}.log \
    ivector-compute-plda ark:data/voxceleb1_train_filtered_trainNdev/spk2utt \
    "ark:ivector-subtract-global-mean ark:dump/voxceleb1_train_filtered_trainNdev/emb.${embname} ark:- | transform-vec dump/voxceleb1_train_filtered_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    dump/voxceleb1_train_filtered_trainNdev/plda.${embname} || exit 1;
fi
# stage 9
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 7: Calculate score"
  # Get results using the out-of-domain PLDA model.
  $train_cmd exp/scores/log/spkid_scoring_${embname}.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 dump/voxceleb1_train_filtered_trainNdev/plda.${embname} - |" \
    "ark:ivector-subtract-global-mean dump/voxceleb1_train_filtered_trainNdev/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec dump/voxceleb1_train_filtered_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean dump/voxceleb1_train_filtered_trainNdev/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec dump/voxceleb1_train_filtered_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat data/${eval_set}/trials | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_${eval_set}_embname${embname} || exit 1;
fi
# stage 10
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "stage 8: EER & MinDCF"
  qsub -cwd -o log/qsub.run.eer_mindcf.voxceleb1.${embname}.log -e log/qsub.run.eer_mindcf.voxceleb1.${embname}.log -l mem_free=200G,ram_free=200G run.eer_mindcf.voxceleb1.sh --eval_set ${eval_set} --embname ${embname}
fi
### JJ - end
