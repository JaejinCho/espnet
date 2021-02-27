#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script add key (uttid), value (e.g. link for ark) pairs

# Begin configuration section.
verbose=0
loc=input
k=label

if [ -f path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh || exit 1;
set -e
set -u

if [ $# != 2 ]; then
    echo "Usage: $0 [options] <json> <data>";
    echo "e.g. : $0 dump/train/data.json exp/xvector_nnet_1a//xvectors_train/xvector.scp"
    echo "e.g. : $0 --loc output -k spklab dump/train/data.json data/train/utt2spklab"
    echo "Options: "
    echo "  --loc <input|output> # location when adding <data> to <json> "
    echo "  --k <e.g. spkint> # k used for actual data values. This is used when your <data> is NOT .scp file"
    exit 1;
fi

json=$1
data=$2
dir=$(dirname ${json})
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
rm -f ${tmpdir}/*.scp

# case 1: if <data> is .scp file
# feats scp
if [ $(echo ${data} | awk -F'.' '{print $NF}') == scp ]; then
    cat ${data} > ${tmpdir}/feat.scp

    if [ ${loc} == output ];then
        dim_loc=odim
    else
        dim_loc=idim
    fi
    # ${dim_loc}.scp
    touch ${tmpdir}/${dim_loc}.scp
    idim=$(copy-vector --print-args=false scp:${tmpdir}/feat.scp ark,t:- | head -n 1 | wc -w)
    idim=$(( idim - 3 ))
    cat ${tmpdir}/feat.scp | awk '{print $1 " " '"${idim}"'}' > ${tmpdir}/${dim_loc}.scp

    # convert to json
    rm -f ${tmpdir}/*.json
    for x in ${tmpdir}/feat.scp ${tmpdir}/${dim_loc}.scp; do
        k=`basename ${x} .scp` # JJ: What does this do? ANS: this removes .scp in the end from the simple "basename ${x}" output
        cat ${x} | scp2json.py --key ${k} > ${tmpdir}/${k}.json
    done

    # add to json
    loc_opt=
    if [ ${loc} == output ];then
        loc_opt="--is-input False"
    fi
    addjson.py ${loc_opt} --verbose ${verbose} \
        ${json} ${tmpdir}/feat.json ${tmpdir}/${dim_loc}.json > ${tmpdir}/data.json
# case 2: if <data> is not scp but other type of files
else
    cat ${data} > ${tmpdir}/data.txt

    # convert to json
    rm -f ${tmpdir}/*.json
    cat ${tmpdir}/data.txt | scp2json.py --key ${k} > ${tmpdir}/${k}.json

    # add to json
    loc_opt=
    if [ ${loc} == output ];then
        loc_opt="--is-input False"
    fi
    addjson.py ${loc_opt} --verbose ${verbose} \
        ${json} ${tmpdir}/${k}.json > ${tmpdir}/data.json
fi

mkdir -p ${dir}/.backup
echo "json updated. original json is kept in ${dir}/.backup."
cp ${json} ${dir}/.backup/$(basename ${json})
cp ${tmpdir}/data.json ${json}
rm -rf ${tmpdir}

