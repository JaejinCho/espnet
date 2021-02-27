#!/bin/bash
# Copyright 2020   Jaejin Cho (*** NOT USEABLE *** Copied and edited from
# /export/b11/jcho/kaldi_20190316/egs/sre10/v1/local/cosine_scoring.sh)
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script trains an LDA transform and does cosine scoring.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 <enroll-ivec-dir> <test-ivec-dir> <trials-file> <scores-dir>"
  echo "e.g., local/cosine_scoring.sh exp/ivectors_sre10_train exp/ivectors_sre10_test $trials exp/scores_gmm_2048_ind_pooled"
fi

enroll_ivec_dir=$1
test_ivec_dir=$2
trials=$3
scores_dir=$4

mkdir -p $scores_dir/log
run.pl $scores_dir/log/cosine_scoring.log \
  cat $trials \| awk '{print $1" "$2}' \| \
 ivector-compute-dot-products - \
  scp:${enroll_ivec_dir}/spk_ivector.scp \
  "ark:ivector-normalize-length scp:${test_ivec_dir}/ivector.scp ark:- |" \
   $scores_dir/cosine_scores || exit 1;

## writing
 cat $trials | awk '{print $1" "$2}' | \
 ivector-compute-dot-products - \
 "ark:ivector-normalize-length ark:dump/${eval_set}/emb.${embname} ark:- |" \
 "ark:ivector-normalize-length ark:dump/${eval_set}/emb.${embname} ark:- |" \
 exp/cs_scores_voxceleb1_${eval_set}_embname${embname} || exit 1;
##


#    ivector-plda-scoring --normalize-length=true \
#    "ivector-copy-plda --smoothing=0.0 dump/voxceleb1_train_filtered_trainNdev/plda.${embname} - |" \
#    "ark:ivector-subtract-global-mean dump/voxceleb1_train_filtered_trainNdev/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec dump/voxceleb1_train_filtered_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#    "ark:ivector-subtract-global-mean dump/voxceleb1_train_filtered_trainNdev/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec dump/voxceleb1_train_filtered_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#    "cat data/${eval_set}/trials | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_${eval_set}_embname${embname} || exit 1;
