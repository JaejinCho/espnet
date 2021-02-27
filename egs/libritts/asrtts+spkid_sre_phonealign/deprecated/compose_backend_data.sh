#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

# JJ: *** THIS WILL BE TO BE REMOVED IF /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1/compose_backend_data.sh WORKS
# Edited from /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1/run_030_extract_xvectors.sh

. ./cmd.sh
. ./path.sh
set -e

embname=sre20

. parse_options.sh || exit 1;

xvector_dir=exp/xvectors/${embname}
mkdir -p ${xvector_dir}
echo "xvector data directory for backend: ${xvector_dir}"

if [ $stage -le 4 ]; then
    # merge eval x-vectors lists
    mkdir -p $xvector_dir/sre16_eval40_yue
    cat dump/sre16_eval40_yue_{enroll,test}_evaluation/embavg_${embname}/xvector.scp > $xvector_dir/sre16_eval40_yue/xvector.scp
    mkdir -p $xvector_dir/sre16_eval40_tgl
    cat dump/sre16_eval40_tgl_{enroll,test}_evaluation/embavg_${embname}/xvector.scp > $xvector_dir/sre16_eval40_tgl/xvector.scp
    mkdir -p $xvector_dir/sre19_eval_cmn2
    cat dump/sre19_eval_{enroll,test}_cmn2_evaluation/embavg_${embname}/xvector.scp > $xvector_dir/sre19_eval_cmn2/xvector.scp
    mkdir -p $xvector_dir/sre20cts_eval
    cat dump/sre20cts_eval_{enroll,test}_evaluation/embavg_${embname}/xvector.scp > $xvector_dir/sre20cts_eval/xvector.scp
fi

if [ $stage -le 6 ];then
    # merge datasets and x-vector list for plda training
    if [ ! -d data/sre16-8 ]; then
        utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
        			  data/sre16-8 \
        			  data/sre16_train_dev_cmn data/sre16_train_dev_ceb \
        			  data/sre16_eval_tr60_yue data/sre16_eval_tr60_tgl \
        			  data/sre18_cmn2_train_lab
    fi
    mkdir -p $xvector_dir/sre16-8
    cat $xvector_dir/{sre16_train_dev_cmn,sre16_train_dev_ceb,sre16_eval_tr60_yue,sre16_eval_tr60_tgl,sre18_cmn2_train_lab}/xvector.scp \
    	> $xvector_dir/sre16-8/xvector.scp

    utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    			  data/realtel_noeng \
    			  data/sre16-8 data/fisher_spa
    mkdir -p $xvector_dir/realtel_noeng
    cat $xvector_dir/{sre16-8,fisher_spa}/xvector.scp > $xvector_dir/realtel_noeng/xvector.scp
  
    utils/combine_data.sh --extra-files "utt2num_frames utt2lang" \
    			  data/realtel_alllangs data/sre_tel data/realtel_noeng
    mkdir -p $xvector_dir/realtel_alllangs
    cat $xvector_dir/{sre_tel,realtel_noeng}/xvector.scp > $xvector_dir/realtel_alllangs/xvector.scp
fi

exit
