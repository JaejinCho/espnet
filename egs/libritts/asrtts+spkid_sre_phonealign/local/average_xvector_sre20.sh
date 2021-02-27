#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

embname="" # e.g.) sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup
. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


if [ ! -n "${embname}" ]; then
    echo "Give a proper string to embname parameter"
else
    echo "Embedding name (model conf.) in xvector extraction: ${embname}"
fi

# for training
for corpus_name in sre_tel fisher_spa; do
    echo "Averaging for train corpus ..."
    bash average_xvector.sh --random true dump/${corpus_name}_{train,dev}/emb.${embname} 2>&1 | tee log/average_xvector.train.${corpus_name}.${embname}.log
done
# for adaptation
for corpus_name in sre16_train_dev_cmn sre16_train_dev_ceb sre16_eval_tr60_yue sre16_eval_tr60_tgl sre18_cmn2_train_lab; do
    echo "Averaging for adapt corpus ..."
    bash average_xvector.sh dump/${corpus_name}_{train,dev}/emb.${embname} 2>&1 | tee log/average_xvector.adapt.${corpus_name}.${embname}.log
done
# for evaluation
for corpus_name in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test; do
    echo "Averaging for eval corpus ..."
    bash average_xvector.sh dump/${corpus_name}_evaluation/emb.${embname} 2>&1 | tee log/average_xvector.evaluation.${corpus_name}.${embname}.log
done
