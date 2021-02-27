#!/bin/bash

# JJ: Edited from /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/run_backendonly_cpuparallel.voxceleb1.sh
# JJ: For ESPnet expriments, I can simply change model_path to a new one in input
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

# config files
#train_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
#model=model.loss.best # or give just a specific snapshot (without any path info. e.g. snapshot.ep.6)
model_path="" # e.g., exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Speaker id related
#spkloss_weight=0

# Speaker id backend related
#expdir=""
lda_dim=150

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/60

# exp tag
tag="" # tag for managing experiments.

# corpus for spkemb extraction
corpus_name=""

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`

echo Embedding name: ${embname}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    pids=() # initialize pids
    # decode in parallel with cpus
    for extract_set in ${corpus_name}_train ${corpus_name}_dev; do
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
            --model ${model_path} \
            --config ${decode_config} \
            --feat-scp ${splitdir}/feats.JOB.scp \
            --out-file ${outemb}.spltixJOB # if files with same names exist, files are overwritten from the beginning

        #cat ${outemb}.* | sort -k1 > ${outemb}
        #cat ${outemb}.spltix* | sort -k1 > ${outemb} # cat does NOT cause error for catting a to a large file. it might be the sort part so I added "-T" option to it as below to avoid space (I think memory) error
        cat ${outemb}.spltix* | sort -k1 -T tmp_sort/ > ${outemb}
        rm ${outemb}.spltix*
        # (TODO: Remove ${outemb}.*)
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
