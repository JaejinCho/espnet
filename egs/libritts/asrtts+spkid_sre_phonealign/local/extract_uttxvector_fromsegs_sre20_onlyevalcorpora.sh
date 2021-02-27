#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

model_path="" # e.g.) exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1
nj=70
#embname="" # e.g.) sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup
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
# for evaluation
for corpus_name in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test; do
    echo "Extract utt-wise xvectors from concated segment-wise feature sequences of the corresponding utterances for eval corpus: ${corpus_name} ..."
    bash extract_uttxvector_fromsegs_parallel.sh --nj ${nj} --model_path ${model_path} dump/${corpus_name}_evaluation/feats.scp 2>&1 | tee log/extract_uttxvector_fromsegs.evaluation.${corpus_name}.${embname}.log
done
