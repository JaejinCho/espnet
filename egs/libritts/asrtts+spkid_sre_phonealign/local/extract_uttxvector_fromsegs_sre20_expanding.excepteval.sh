#!/bin/bash
# Edited from local/extract_uttxvector_fromsegs_sre20.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

model_path="" # e.g.) exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1
#embname="" # e.g.) sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup
nj=70

# Feature extraction/normalization related
feat_dir=/export/b18/jcho/espnet3/egs/libritts/tts_featext
eval_only=false # true or false
eval_seg_NOdiscard=true # true or false

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


if [ ! -n "${model_path}" ]; then
    echo "Give a proper string to model_path parameter"
else
    echo "Model conf. in xvector extraction: ${model_path}"
fi

embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
echo "Embedding name (model conf.) in xvector extraction: ${embname}"

if [ "${eval_only}" == false ]; then
    echo "for training corpora"
    for corpus_name in sre_tel fisher_spa; do
        echo "Extract utt-wise xvectors from concated segment-wise feature sequences of the corresponding utterances for training corpus: ${corpus_name} ..."
        bash extract_uttxvector_fromsegs_parallel.sh --nj ${nj} --model_path ${model_path} --random true dump/${corpus_name}_{train,dev}/feats.scp 2>&1 | tee log/extract_uttxvector_fromsegs.train.${corpus_name}.${embname}.log
    done
fi

if [ "${eval_only}" == false ]; then
    echo "for adaptation corpora"
    for corpus_name in sre16_train_dev_cmn sre16_train_dev_ceb sre16_eval_tr60_yue sre16_eval_tr60_tgl sre18_cmn2_train_lab; do
        echo "Extract utt-wise xvectors from concated segment-wise feature sequences of the corresponding utterances for adapt corpus: ${corpus_name} ..."
        bash extract_uttxvector_fromsegs_parallel.sh --nj ${nj} --model_path ${model_path} dump/${corpus_name}_{train,dev}/feats.scp 2>&1 | tee log/extract_uttxvector_fromsegs.adapt.${corpus_name}.${embname}.log
    done
fi
