Outline: "Only for backend related command runs"

# TODO
# # Expand to NOdiscard


# list of corpora for x-vector extraction: ones within parentheses are not used for now. Corpora for training & adapting are divided into *{_train,_dev} in dumping while one for evaluation has suffix *_evaluation in dumping
# # for training LDA/PLDA: 2 corpora (+1)
sre_tel (cncelebcat_tel) fisher_spa
# # for adapting LDA/PLDA: 5 corpora (+1)
sre16_train_dev_cmn sre16_train_dev_ceb sre16_eval_tr60_yue sre16_eval_tr60_tgl sre18_cmn2_train_lab (sre18_dev_unlabeled)
# # for evaluation (require: both FE and x-vector extraction): 8 corpora
sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test
# # # ***** for sre19_eval_enroll_cmn2 & sre19_eval_test_cmn2, I generate data dir again for them with "_NOdiscard" as their suffices, NOT to discard any segments for later steps in back-end

# < option 1: utt-wise averaged x-vector extraction > (TODO: the 3 blocks (for spkid only, spkid + tts, and spkid + tts semi) below did not run yet for checking the code. Check by running it and if it works, change the name to local/average_xvector_sre20.sh and move the original one to deprecated)
# # ***** Run the command below after setting model_path & embname properly
bash local/average_xvector_sre20_expanding.sh --model_path ${model_path} 2>&1 | tee log/average_xvector_sre20.${embname}.log
# # << spkid only >>
model_path=exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup/results/model.loss.best
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# # << spkid + tts >>
# # # snapshot.ep.18
model_path=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_unispksampling_lr3e-4_spkloss_weight0.03_mem35G_ijson_sup/results/snapshot.ep.18
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# # << spkid + tts semi >>
# # # snapshot.ep.1
model_path=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1 # dir name was too long to process later parts below so I changed copy it with shorter name
expdirname=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G
expdirname_shorter=exp/sre20_semi_transferfromsup
cp -r ${expdirname} ${expdirname_shorter}
model_path=${expdirname_shorter}/results/snapshot.ep.1
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`

# < option 2: utt-wise xvector extraction from cated seg-wise xvectors >
# # Description: Get one xvector per utterance: First, concatenate the segments from the utterance and get one xvector for the utterance
# # (TODO: do normalization after segment concatenation)
# # ***** Run the command below after setting model_path & embname properly
bash local/extract_uttxvector_fromsegs_sre20_expanding.sh --nj 70 --model_path ${model_path} --eval_seg_NOdiscard false 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.log
# # << spkid only >> (TODO: ING. I kept the previous extracted xvectors by "for fname in `find dump -name "*sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup_model.loss.best*"`;do mv $fname ${fname}_old; done")
# # # 50 ep best model
model_path=exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup/results/model.loss.best
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# # # # with NOdiscard option set
(DONE) bash local/extract_uttxvector_fromsegs_sre20_expanding.sh --nj 70 --model_path ${model_path} --eval_seg_NOdiscard true --eval_only true 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.NOdiscard_evalonly.log
# # # (DONE) 100 ep best model
expdirname=exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_lr3e-4_100ep_ttslossw0_spkloss_weight1_mem35G_ijson_sup
expdirname_shorter=exp/sre20_train_spkidonly_unsync_shufflebatch_multicorpora_unispksample_lr3e-4_100ep_mem35G_ijson_sup
mv ${expdirname} ${expdirname_shorter}
model_path=${expdirname_shorter}/results/model.loss.best
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# # # # with NOdiscard option set
(DONE) bash local/extract_uttxvector_fromsegs_sre20_expanding.sh --nj 70 --model_path ${model_path} --eval_seg_NOdiscard true 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.NOdiscard4eval.log
# # << spkidtts >> (TODO: ING. I kept the previous extracted xvectors by "for fname in `find dump/ -name "*${embname}*" | grep fromcatsegs`;do mv $fname ${fname}_old; done")
# # # ep18 model
model_path=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_unispksampling_lr3e-4_spkloss_weight0.03_mem35G_ijson_sup/results/snapshot.ep.18
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# # # # with NOdiscard option set
(DONE) bash local/extract_uttxvector_fromsegs_sre20_expanding.sh --nj 70 --model_path ${model_path} --eval_seg_NOdiscard true --eval_only true 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.NOdiscard_evalonly.log
# # # (ING) ep2 model w/ kaldiaug
expdirname=/export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sreNkaldiaug_phonealign/exp/sre20Nkaldiaug_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_kaldidataaug_lr1e-4_transfer_from_sup_ep18_notrainerupdate_spkloss_weight0.03_fullyunsync_cleanrecon_mem55G_ijson_sup
expdirname_shorter=/export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sreNkaldiaug_phonealign/exp/sre20Nkaldiaug_sup_transferfromep18sup_spkloss_weight0.03
cp -r ${expdirname} ${expdirname_shorter} # cp since the training is still going on
model_path=${expdirname_shorter}/results/snapshot.ep.3
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# # # # with NOdiscard option set
(ING) bash local/extract_uttxvector_fromsegs_sre20_expanding.sh --nj 70 --model_path ${model_path} --eval_seg_NOdiscard true 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.NOdiscard.log
# # << spkidtts - semi >>
# # # transfer from ep14 sup model
model_path=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1 # dir name was too long to process later parts below so I changed copy it with shorter name
expdirname=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G
expdirname_shorter=exp/sre20_semi_transferfromsup
cp -r ${expdirname} ${expdirname_shorter}
model_path=${expdirname_shorter}/results/snapshot.ep.1
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# # # # with NOdiscard option set
(DONE) bash local/extract_uttxvector_fromsegs_sre20_expanding.sh --nj 70 --model_path ${model_path} --eval_seg_NOdiscard true --eval_only true 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.NOdiscard_evalonly.log
# # # (DONE) transfer from ep18 sup model w/ lr scheduler
expdirname=exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr1e-4Nscheduler_transfer_from_supervised_ep18_currentbest_notrainerupdate_spkloss_weight0.03_fullyunsync_mem50G_ijson
expdirname_shorter=exp/sre20_semi_transferfrom_ep18sup_w_lrscheduler
cp -r ${expdirname} ${expdirname_shorter} # cp instead of mv since the training is still going on
model_path=${expdirname_shorter}/results/snapshot.ep.7
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
# # # # with NOdiscard option set
(DONE) bash local/extract_uttxvector_fromsegs_sre20_expanding.sh --nj 70 --model_path ${model_path} --eval_seg_NOdiscard true 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.NOdiscard4eval.log


## 3. (RUN THIS at /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1) Combine data (similar to the process left after x-vector extraction in run_030_extract_xvectors.sh under /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1/) and run backend training and evaluation
## (DONE) < spkid only >: Refer to /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1/list_run.sh
## (DONE) < spkidtts >
## (DONE) < spkidtts - semi >

## 4. (RUN THIS at /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1) backend training and evaluation
## (ING) < spkid only >: Refer to /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1/list_run.sh
