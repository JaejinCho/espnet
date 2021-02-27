#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=1000
random=false


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# 1. If applicable, WITH sort, cat dump/[corpus name]_{train,dev}/emb.[embname] > dump/[corpus name]_trainNdev/emb.[embname] && rm dump/[corpus name]_{train,dev}/emb.[embname]
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ $# -eq 2 ]; then
    	# this is when a corpus embeddings are separated into _train and _dev for dump for training for front-end
        echo "Stage 0: Cat $@ and sort by the first column. Then, remove them after the process."
        dname=`dirname $1 | rev | cut -d'_' -f2- | rev`
        dname_trainNdev=${dname}_trainNdev
        embname=`basename $1 | cut -d'.' -f2-`
           
        mkdir -p ${dname_trainNdev}
        ori_emb_ark=${dname_trainNdev}/emb.${embname}
        # for emb
        if [ ! -f ${ori_emb_ark} ]; then
            cat $@ | sort -k1 -T tmp_sort/ > ${ori_emb_ark} && rm $@ # -T option to avoid the "No space *" error
        else
            echo "${ori_emb_ark} is already prepared. Skip cat and sort"
        fi
        # for utt2num_frames (this stays the same regardless of extractor)
    	if [ ! -f ${dname_trainNdev}/utt2num_frames ]; then
        	cat ${dname}_{train,dev}/utt2num_frames | sort -k1 -T tmp_sort/ > ${dname_trainNdev}/utt2num_frames
        else
            echo "${dname_trainNdev}/utt2num_frames is already prepared. Skip cat and sort"
    	fi
        echo "Catted and sorted emb at ${ori_emb_ark}"
    elif [ $# -eq 1 ]; then
    	# this is when a corpus embedding is for evaluation
        ori_emb_ark=$1
    else
        echo "# positional arguments should be 1 or 2." && exit 1
    fi
fi

# 2. Average x-vectors and make xvector.scp
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    dname_ori=`dirname ${ori_emb_ark}`
    embname=`basename ${ori_emb_ark} | cut -d'.' -f2-`
	dname_out=${dname_ori}/embavg_${embname}
    mkdir -p ${dname_out}
    echo "Stage 1: Processing with ${ori_emb_ark} to average. Writing output: xvector.ark.txt, xvector.ark and xvector.scp into ${dname_out}/"
    if [ "$random" = true ]; then # this condition is the same as [ $# -eq 2 ]
        average_xvector.py --random ${dname_ori}/utt2num_frames ${ori_emb_ark} ${dname_out}/xvector.ark.txt
    else
        average_xvector.py ${ori_emb_ark} ${dname_out}/xvector.ark.txt
    fi
    copy-vector ark:${dname_out}/xvector.ark.txt ark,scp:${dname_out}/xvector.ark,${dname_out}/xvector.scp
    echo "Finish generating averaged xvector"
fi
