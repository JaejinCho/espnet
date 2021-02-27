#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

maxframes=2000
minframes=10
nlsyms=""
no_feat=false

help_message="usage: $0 olddatadir newdatadir"

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

sdir=$1
odir=$2
mkdir -p ${odir}/tmp

if [ ${no_feat} = true ]; then
    # for machine translation
    cut -d' ' -f 1 ${sdir}/text > ${odir}/tmp/reclist1
else
    echo "extract utterances having less than $maxframes or more than $minframes frames"
    utils/data/get_utt2num_frames.sh ${sdir}
    < ${sdir}/utt2num_frames  awk -v maxframes="$maxframes" '{ if ($2 < maxframes) print }' \
        | awk -v minframes="$minframes" '{ if ($2 > minframes) print }' \
        | awk '{print $1}' > ${odir}/tmp/reclist1
fi

reduce_data_dir.sh ${sdir} ${odir}/tmp/reclist1 ${odir}
utils/fix_data_dir.sh ${odir}

oldnum=$(wc -l ${sdir}/feats.scp | awk '{print $1}')
newnum=$(wc -l ${odir}/feats.scp | awk '{print $1}')
echo "change from $oldnum to $newnum"
