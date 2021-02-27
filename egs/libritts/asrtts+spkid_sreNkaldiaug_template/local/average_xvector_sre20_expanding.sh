#!/bin/bash

# Edited from local/average_xvector_sre20.sh to include segment-wise embedding
# extractions before getting utterance-wise averaged embedding from them (TODO: Once finished writing the
# code, replace the original code)
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

model_path="" # e.g.) exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1
#expdir="" # e.g.) exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup
#model=model.loss.best
stage=0
stop_stage=100

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
echo "Embedding name (model conf.) in xvector extraction: ${embname}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: getting segment-wise x-vectors"
    # # for training & adpatation
    for corpus_name in sre_tel fisher_spa sre16_train_dev_cmn sre16_train_dev_ceb sre16_eval_tr60_yue sre16_eval_tr60_tgl sre18_cmn2_train_lab; do
        bash extract_spkemb_cpuparallel.labcorpus.sh --model_path ${model_path} --corpus_name ${corpus_name} --nj 70 --ngpu 0 2>&1 | tee log/extract_spkemb.${embname}.sre20.${corpus_name}.log &
    done
    # # for evaluation
    for corpus_name in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test; do
        bash extract_spkemb_cpuparallel.evalcorpus.sh --model_path ${model_path} --corpus_name ${corpus_name} --nj 70 --ngpu 0 2>&1 | tee log/extract_spkemb.${embname}.sre20.${corpus_name}.log &
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: getting averaged utt-wise x-vectors from segment-wise xvectors"
    # # for training
    for corpus_name in sre_tel fisher_spa; do
        echo "Averaging for train corpus ..."
        bash average_xvector.sh --random true dump/${corpus_name}_{train,dev}/emb.${embname} 2>&1 | tee log/average_xvector.train.${corpus_name}.${embname}.log
    done
    # # for adaptation
    for corpus_name in sre16_train_dev_cmn sre16_train_dev_ceb sre16_eval_tr60_yue sre16_eval_tr60_tgl sre18_cmn2_train_lab; do
        echo "Averaging for adapt corpus ..."
        bash average_xvector.sh dump/${corpus_name}_{train,dev}/emb.${embname} 2>&1 | tee log/average_xvector.adapt.${corpus_name}.${embname}.log
    done
    # # for evaluation
    for corpus_name in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test; do
        echo "Averaging for eval corpus ..."
        bash average_xvector.sh dump/${corpus_name}_evaluation/emb.${embname} 2>&1 | tee log/average_xvector.evaluation.${corpus_name}.${embname}.log
    done
fi
