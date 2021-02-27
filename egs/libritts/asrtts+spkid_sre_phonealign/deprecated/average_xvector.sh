#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=1000
random=false


. utils/parse_options.sh || exit 1;

# 1. If applicable, WITH sort, cat dump/[corpus name]_{train,dev}/emb.[embname] > dump/[corpus name]_trainNdev/emb.[embname] && rm dump/[corpus name]_{train,dev}/emb.[embname]
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ $# -eq 2 ]; then
    	# this is when a corpus embeddings are separated into _train and _dev for dump for training for front-end
        echo "Cat $@ and sort by the first column. Then, remove them after the process."
        dname=`dirname $1 | rev | cut -d'_' -f2- | rev`
        dname_trainNdev=${dname}_trainNdev
        embname=`basename $1 | cut -d'.' -f2-`
    
        mkdir -p ${dname_trainNdev}
        # for emb
        ori_emb_ark=${dname_trainNdev}/emb.${embname}
        cat $@ | sort -k1 -T tmp_sort/ > ${ori_emb_ark} && rm $@ # -T option to avoid the "No space *" error
        # for utt2num_frames (this stays the same regardless of extractor)
    	if [ ! -f ${dname_trainNdev}/utt2num_frames ]; then
        	cat ${dname}_{train,dev}/utt2num_frames | sort -k1 -T tmp_sort/ > ${dname_trainNdev}/utt2num_frames
    	fi
        echo "Catted and sorted emb at ${ori_emb_ark}"
    else
    	# this is when a corpus embedding is for evaluation
        ori_emb_ark=$1
    fi
fi

# 2. Average x-vectors and make xvector.scp
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    dname_ori=`dirname ${ori_emb_ark}`
    embname=`basename ${ori_emb_ark} | cut -d'.' -f2-`
	dname_out=${dname_ori}/embavg_${embname}
    mkdir -p ${dname_out}
    echo "Processing with ${ori_emb_ark} to average. Writing output: xvector.ark.txt, xvector.ark and xvector.scp into ${dname_out}/"
    if [ "$random" = true ]; then # this condition is the same as [ $# -eq 2 ]
        python average_xvector.py --random ${dname_trainNdev}/utt2num_frames ${ori_emb_ark} ${dname_out}/xvector.ark.txt
    else
        python average_xvector.py ${ori_emb_ark} ${dname_out}/xvector.ark.txt
    fi
    copy-vector ark:${dname_out}/xvector.ark.txt ark,scp:${dname_out}/xvector.ark,xvector.scp
fi
