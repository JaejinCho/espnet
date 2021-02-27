#!/bin/bash

# Edited from utils/extract_uttxvector_fromsegs.sh to include parallel cpu processings

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=1000
random=false
model_path="" # e.g.) exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1
nj=70

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

datadir=data/$(basename ${dname_ori} | rev | cut -d'_' -f2- | rev)
f_utt2spk=${datadir}/utt2spk
num_spk=$(wc -l ${datadir}/spk2utt | awk '{print $1}')
if [ ${num_spk} -lt ${nj} ]; then
    nj=${num_spk}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	dname_out=${dname_ori}/emb_fromcatsegs_${embname}
    mkdir -p ${dname_out}

    echo "Stage 1: Processing with ${dname_ori}/feats.scp to get an utt-wise xvector per the concated segment-wise featseqs of the utt. Writing output: xvector.ark.txt, xvector.ark and xvector.scp into ${dname_out}/"

    feats_scp=${dname_ori}/feats.scp
    splitdir_feats=${dname_ori}/feats_split${nj}/
    if [ ! -d ${splitdir_feats} ]; then
        mkdir -p ${splitdir_feats}
        split_scps=""
        for n in $(seq $nj);do
            split_scps="${split_scps} ${splitdir_feats}/feats.${n}.scp"
        done
        utils/split_scp.pl --utt2spk=${f_utt2spk} $feats_scp $split_scps || exit 1
    fi

    # set option to be given as arguments to the main python script
    option="--model ${model_path} ${splitdir_feats}/feats.JOB.scp ${dname_out}/xvector.ark.JOB.txt"
    if [ "$random" = true ]; then # this condition is the same as [ $# -eq 2 ]

        utt2num_frames_scp=${dname_ori}/utt2num_frames
        splitdir_utt2num_frames=${dname_ori}/utt2num_frames_split${nj}
        if [ ! -d ${splitdir_utt2num_frames} ]; then
            mkdir -p ${splitdir_utt2num_frames}
            for ix in $(seq $nj); do
                filter_scp.pl ${splitdir_feats}/feats.${ix}.scp ${utt2num_frames_scp} > ${splitdir_utt2num_frames}/utt2num_frames.${ix}.split
            done
        fi
        option="--random ${splitdir_utt2num_frames}/utt2num_frames.JOB.split ${option}"
        echo "Random selection of segments is ON (intead of using all segments)"
    fi

    # parallel run on cpus
    ${train_cmd} JOB=1:${nj} ${dname_out}/log/extract_uttxvector_fromsegs.JOB.log \
    extract_uttxvector_fromsegs.py ${option}

    # put split xvectors together and delete the splits
    cat ${dname_out}/xvector.ark.*.txt | sort -k1 -T tmp_sort/ > ${dname_out}/xvector.ark.txt
    rm -rf ${dname_out}/xvector.ark.*.txt
    # Change the format to be used in Jesus' codes
    copy-vector ark:${dname_out}/xvector.ark.txt ark,scp:${dname_out}/xvector.ark,${dname_out}/xvector.scp

    echo "Finish generating utt-wise xvectors from the concated seg-wise featseqs"
fi
