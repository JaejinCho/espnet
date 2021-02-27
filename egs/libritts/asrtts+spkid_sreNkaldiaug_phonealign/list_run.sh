mkdir -p log
ln -s /export/c06/jcho/espnet3/egs/libritts/asrtts+spkid_sreNkaldiaug_phonealign/dump dump

# < exp >: clean speech reconstruction from noisy/clean speech input
# 1. data prep (TODO: check resulting outputs from every stage are correct)
## stage 0
bash run.cleanrecon.sh --stage 0 --stop_stage 0 2>&1 | tee log/run.cleanrecon.stage0.log
## stage 1 and 2 at the same time (this is rerun before sleep since from the first run, some of files in dump directory were not sorted and raised some errors.)
(DONE: error) bash run.cleanrecon.sh --stage 1 --stop_stage 2 2>&1 | tee log/run.cleanrecon.stage1N2.log # error alreday from stage 1 for 5 corpora as below
(DONE) bash run.cleanrecon.stage1_error_6corpora_sre16_train_dev_cmn_babel_lithuanian_babel_javanese_babel_telugu_babel_pashto_sre16_eval_tr60_tgl.sh --stage 1 --stop_stage 1 2>&1 | tee log/run.cleanrecon.stage1_error_6corpora_sre16_train_dev_cmn_babel_lithuanian_babel_javanese_babel_telugu_babel_pashto_sre16_eval_tr60_tgl.log # rerun for 5 copora that did not run properly for only stage 1
(DONE) bash run.cleanrecon.sh --stage 2 --stop_stage 2 2>&1 | tee log/run.cleanrecon.stage2.log # This seemed to finish well by checking the results although it did not print out 'echo "stage 2 Finished"' for some reason. So I stopped the job manually once checking there is no qsub job regarding this. * Note: related log files are possibly appended rather than overwritten. That could be the reason they include some error messages.
## stage 3
(DONE) bash run.cleanrecon.sh --stage 3 --stop_stage 3 2>&1 | tee log/run.cleanrecon.stage3.log
## (DONE) stage additional
### remove textNtoken in data.json files --- start
##### nolab corpora
for cname in $corpora_list_nolab;do ./remove_textNtoken_indatajson.py dump/${cname}_combined_nolab/data.cleanrecon.json dump/${cname}_combined_nolab/data.cleanrecon.wo_textNtoken.json; done
##### lab corpora
for cname in $corpora_list_lab;do ./remove_textNtoken_indatajson.py dump/${cname}_combined_train/data.cleanrecon.json dump/${cname}_combined_train/data.cleanrecon.wo_textNtoken.json; ./remove_textNtoken_indatajson.py dump/${cname}_combined_dev/data.cleanrecon.json dump/${cname}_combined_dev/data.cleanrecon.wo_textNtoken.json; done
### remove textNtoken in data.json files --- end
## stage 4
### (TODO) fully supervised spkid
### fully supervised tts + spkid
(ING, I think this should run again with conf/*kaldidataaug.yaml for train_config) bash run.cleanrecon.sup.sh --mem_request 50G --ngpu 1 --spkloss_weight 0.03 --stage 4 --stop_stage 4 2>&1 | tee log/run.cleanrecon.sup.stage4.spkloss_weight0.03.mem50G.log
(ING) bash run.cleanrecon.sup.sh --mem_request 55G --resume_trainerupdate False --tag transfer_from_sup_ep18_notrainerupdate --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_kaldidataaug_lr1e-4.yaml --resume ../asrtts+spkid_sre_phonealign/exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_unispksampling_lr3e-4_spkloss_weight0.03_mem35G_ijson_sup/results/snapshot.ep.18 --stage 4 --stop_stage 4 2>&1 | tee log/run.cleanrecon.sup.transfer_from_sup_ep18_notrainerupdate.log
### (TODO) semi-supervised tts + spkid
#(TODO) bash run.cleanrecon.semisup.sh --mem_request 50G --ngpu 1 --spkloss_weight 0.03 --stage 4 --stop_stage 4 2>&1 | tee log/run.cleanrecon.sup.stage4.spkloss_weight0.03.log
1. save dict into shelve db (in qlogin. This this it does not matter if it is k80 or 1080ti)
(DONE) bash run.cleanrecon.semisup.shelve.qlogin.sh --save_shelve_dir shelve_cleanrecon_semi/ --stage 4 --stop_stage 4 --gpu_ix $(free-gpu)
2. train from the shelve db (on 1080ti)
(ING) bash run.cleanrecon.semisup.shelve.sh --mem_request 20G --shelve_train_batchset shelve_cleanrecon_semi/train_batchset.shelve --shelve_valid_batchset shelve_cleanrecon_semi/valid_batchset.shelve --ngpu 1 --spkloss_weight 0.03 --stage 4 --stop_stage 4 2>&1 | tee log/run.cleanrecon.sup.stage4.spkloss_weight0.03.log

### below run might have run while including some error or missing processes. so run them together from the above after removing dump/ and ${dict}
## stage 1
bash run.cleanrecon.sh --stage 1 --stop_stage 1 2>&1 | tee log/run.cleanrecon.stage1.log # some grid errors (IDK what exactly it is but I might be I submit a lot of jobs at once)
bash run.cleanrecon.stage1_error_3corpora_babel_dholuo_babel_guarani_babel_turkish.sh --stage 1 --stop_stage 1 2>&1 | tee log/run.cleanrecon.stage1_error_3corpora_babel_dholuo_babel_guarani_babel_turkish.log # only 3 corpora that raised errors above
## stage 2 (stage 2-1 and 2-2 were all qsubed so I am not sure if they were done in order (supposed to be to be right). I may separate them in to different stage)
bash run.cleanrecon.sh --stage 2 --stop_stage 2 2>&1 | tee log/run.cleanrecon.stage2.log 




# improve training speed (refer to this section in
# ../asrtts+spkid_sre_phonealign/list_run.sh for what TODO next)
