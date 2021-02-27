#!/bin/bash

# Edited from utils/average_xvector.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=1000
random=false
model_path="" # e.g.) exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# 1. If applicable, WITH sort, cat dump/[corpus name]_{train,dev}/feats.scp > dump/[corpus name]_trainNdev/feats.scp
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ $# -eq 2 ]; then
    	# this is when normalized features in a corpus are separated into _train and _dev for dump for training for front-end
        echo "Stage 0: Cat $@ and sort by the first column."
        dname=`dirname $1 | rev | cut -d'_' -f2- | rev`
        dname_trainNdev=${dname}_trainNdev

        mkdir -p ${dname_trainNdev}
        # for dumped feats.scp (this stays the same regardless of extractor)
        if [ ! -f ${dname_trainNdev}/feats.scp ]; then
            cat $@ | sort -k1 -T tmp_sort/ > ${dname_trainNdev}/feats.scp # -T option to avoid the "No space *" error
        else
            echo "${dname_trainNdev}/feats.scp is already prepared. Skip cat and sort"
        fi
        # for utt2num_frames (this stays the same regardless of extractor)
    	if [ ! -f ${dname_trainNdev}/utt2num_frames ]; then
        	cat ${dname}_{train,dev}/utt2num_frames | sort -k1 -T tmp_sort/ > ${dname_trainNdev}/utt2num_frames
        else
            echo "${dname_trainNdev}/utt2num_frames is already prepared. Skip cat and sort"
    	fi
        echo "Catted and sorted {feats.scp,utt2num_frames} at ${dname_trainNdev}/{feats.scp,utt2num_frames}"
    elif [ $# -eq 1 ]; then
    	# this is when normalized features in a corpus is for evaluation
        echo "Nothing to be done for stage 0 with evaluation corpus"
    else
        echo "# positional arguments should be 1 or 2." && exit 1
    fi
fi



# 2. Average x-vectors and make xvector.scp
if [ $# -eq 2 ]; then
    dname_ori=`dirname $1 | rev | cut -d'_' -f2- | rev`_trainNdev
else
    dname_ori=`dirname $1`
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	dname_out=${dname_ori}/emb_fromcatsegs_${embname}
    mkdir -p ${dname_out}
    echo "Stage 1: Processing with ${dname_ori}/feats.scp to get an utt-wise xvector per the concated segment-wise featseqs of the utt. Writing output: xvector.ark.txt, xvector.ark and xvector.scp into ${dname_out}/"
    if [ "$random" = true ]; then # this condition is the same as [ $# -eq 2 ]
        extract_uttxvector_fromsegs.py --random ${dname_ori}/utt2num_frames --model ${model_path} ${dname_ori}/feats.scp ${dname_out}/xvector.ark.txt
    else
        extract_uttxvector_fromsegs.py --model ${model_path} ${dname_ori}/feats.scp ${dname_out}/xvector.ark.txt
    fi
    copy-vector ark:${dname_out}/xvector.ark.txt ark,scp:${dname_out}/xvector.ark,${dname_out}/xvector.scp
    echo "Finish generating utt-wise xvectors from the concated seg-wise featseqs"
fi
