mkdir -p log

# data prep 
# (TODO: DONE) short window cmvn: I think this is very important when using different datasets
bash run.asrttsspkid.sh --stage 0 --stop_stage 0 2>&1 | tee log/run.asrttsspkid.stage0only.log
bash run.asrttsspkid.sh --stage 1 --stop_stage 1 2>&1 | tee log/run.asrttsspkid.stage1only.log
(DONE (one below) but there were some problmes using qsub. It just goes to the next for loop block once qsubs are submitted regardless of the jobs submitted have errors or not. So run the "local/update_json_general.sh" part again)
bash run.asrttsspkid.sh --stage 2 --stop_stage 2 2>&1 | tee log/run.asrttsspkid.stage2only.log
(DONE RERUN: corpor_list excludes babel_{amharic,assamese} sinc they were run manually by me before this rerun)
bash run.asrttsspkid.stage2updatejsononly.sh --stage 2 --stop_stage 2 2>&1 | tee log/run.asrttsspkid.stage2_update_json_general_only.log


# training
(ERROR DUE TO MEMORY ERROR IN READING JSON FILES) bash run.asrttsspkid.sh --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid.stage4only.log
(MEMORY ERROR, AFTER FIX FROM ABOVE) bash run.asrttsspkid_ijson.mem20G.sh --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.mem20G.stage4only.log
(SOME ERROR: refer to train.error.log) bash run.asrttsspkid_ijson.mem30G.sh --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.mem30G.stage4only.log
## the below command is after fix including "popitem" from dicts depending on some condition
(DONE with error: 1) grad norm is nan, 2) mem_killer (using ~40GB memory while the requested one was 30GB)) bash run.asrttsspkid_ijson.mem30G.sh --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.mem30G.stage4only.log
## the below command is after adding uniform sampling of speaker + replace popitem with giving min_batch_size=args.batch_size to make_batchset
### semi-supervised (25 babel nolab corpora + 9 lab corpora)
(MEM KILLER AFTER 9 epoch) bash run.asrttsspkid_ijson.unispksampling.mem30G.sh --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.unispksampling.mem30G.stage4only.log
(MEM KILLER, RESUME from the above: NO improvement) bash run.asrttsspkid_ijson.unispksampling.mem45G.sh --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_spkloss_weight0.03_fullyunsync_mem30G_ijson/results/snapshot.ep.9 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.unispksampling.mem45G.stage4only.resumefromep9.log
(MEM KILLER, from 9 epoch with lr=3e-4 from 1e-3, with self.it=775900) bash run.asrttsspkid_ijson.unispksampling.mem45G.sh --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr3e-4.yaml --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_spkloss_weight0.03_fullyunsync_mem30G_ijson/results/snapshot.ep.9 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.unispksampling.mem45G.stage4only.resumefromep9_lr3e-4from1e-3.log
(ING with 50G mem, from 9 epoch with lr=3e-4 from 1e-3, with self.it=775900) bash run.asrttsspkid_ijson.unispksampling.mem50G.sh --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr3e-4.yaml --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_spkloss_weight0.03_fullyunsync_mem30G_ijson/results/snapshot.ep.9 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.unispksampling.mem50G.stage4only.resumefromep9_lr3e-4from1e-3.log
(ING) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.sh --tag transfer_from_libritts --resume /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_libris_kaldiasr_phonealign_rf3_fs16k/exp/train_train_16k_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_fullyunsync/results/model.loss.best --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.spkloss_weight0.03.onlystage4.transfer_from_libritts_spklossweight0.log
(MEM KILLER) bash run.asrttsspkid_ijson.unispksampling.mem30G.sh --spkloss_weight 0.01 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.unispksampling.mem30G.stage4only.spkloss_weight0.01.log
(MEM KILLER) bash run.asrttsspkid_ijson.unispksampling.mem30G.sh --spkloss_weight 0.003 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.unispksampling.mem30G.stage4only.spkloss_weight0.003.log
#### transfer from well trained supervised model with less lr
(ING with 50G mem, transfer from supervised models 14 epoch (current best considering both spkid and recon metrics) with lr=1e-4 from 3e-4, with self.it=560600) bash run.asrttsspkid_ijson.unispksampling.mem50G.sh --resume_trainerupdate False --tag transfer_from_supervised_ep14_currentbest_notrainerupdate --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr1e-4.yaml --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr3e-4_spkloss_weight0.03_fullyunsync_mem35G_ijson_sup/results/snapshot.ep.14 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson.unispksampling.mem50G.stage4only.transferfrom_supep14currentbest_notrainerupdate_lr1e-4from3e-4.log
(TODO: by increasing the amount of babel data (unlab data) from small amount)
(TODO: just with reduced amount of babel data)

### fully supervised (9 lab corpora)
#### TTS + SPKID (bs64, ep30)
(ING, mem30G causes memory error: RES ~33G) bash run.asrttsspkid_ijson_sup.unispksampling.mem35G.sh --spkloss_weight 0.03 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson_sup.unispksampling.mem35G.stage4only.spkloss_weight0.03.log
(ING spkloss_w=0.01 (decrease), RESUME with self.it=200300 from 5.ep above)  bash run.asrttsspkid_ijson_sup.unispksampling.mem35G.sh --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_spkloss_weight0.03_fullyunsync_mem35G_ijson_sup/results/snapshot.ep.5 --spkloss_weight 0.01 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson_sup.unispksampling.mem35G.stage4only.spkloss_weight0.01.resumefromep5.log
(ING spkloss_w=0.1 (increase), RESUME with self.it=200300 from 5.ep above)  bash run.asrttsspkid_ijson_sup.unispksampling.mem35G.sh --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_spkloss_weight0.03_fullyunsync_mem35G_ijson_sup/results/snapshot.ep.5 --spkloss_weight 0.1 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson_sup.unispksampling.mem35G.stage4only.spkloss_weight0.1.resumefromep5.log
(ING ** cmpr1 ** lr=3e-4, RESUME with self.it=200300 from 5.ep above) bash run.asrttsspkid_ijson_sup.unispksampling.mem35G.sh --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr3e-4.yaml --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_spkloss_weight0.03_fullyunsync_mem35G_ijson_sup/results/snapshot.ep.5 --spkloss_weight 0.03 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson_sup.unispksampling.mem35G.stage4only.spkloss_weight0.03.resumefromep5_lr3e-4from1e-3.log
(ING ** compare this with cmpr1 above ** lr=3e-4, RESUME with self.it=200300 from 5.ep with spkloss_w=0.1 above: This is the same as keeping lr=1e-3 & spkloss_w=0.03 and changing ttsloss=0.3 from 1) bash run.asrttsspkid_ijson_sup.unispksampling.mem35G.sh --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr3e-4.yaml --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_spkloss_weight0.03_fullyunsync_mem35G_ijson_sup/results/snapshot.ep.5 --spkloss_weight 0.1 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson_sup.unispksampling.mem35G.stage4only.spkloss_weight0.1.resumefromep5_lr3e-4from1e-3.log
(ING lr=3e-4, RESUME with self.it=200300 from 7.ep with spkloss_w=0.01 above: finer control of lr) bash run.asrttsspkid_ijson_sup.unispksampling.mem35G.sh --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr3e-4.yaml --resume exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_spkloss_weight0.01_fullyunsync_mem35G_ijson_sup/results/snapshot.ep.7 --spkloss_weight 0.01 --stage 4 --stop_stage 4 2>&1 | tee log/run.asrttsspkid_ijson_sup.unispksampling.mem35G.stage4only.spkloss_weight0.01.resumefromep7_lr3e-4from1e-3.log

#### SPKID only
(DONE) bash run.spkidonly_ijson_sup.unispksampling.mem35G.sh --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --stop_stage 4 2>&1 | tee log/run.spkidonly_ijson_sup.unispksampling.mem35G.stage4only.log
(ING RESUME til 100ep with self.it=1000900 from ep.50 (spklossNacc best)) bash run.spkidonly_ijson_sup.unispksampling.mem35G.sh --train_config conf/train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_lr3e-4_100ep.yaml --resume exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup/results/snapshot.ep.50 --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --stop_stage 4 2>&1 | tee log/run.spkidonly_ijson_sup.unispksampling.mem35G.stage4only.resumefromep50_spklossNaccbest.lrfrom1e-3to3e-4.untilep100.log





# BACK-END (The whole codes are being trimmed for a faster run to list_run.backend.sh)
## 0. list of corpora for x-vector extraction: ones within parentheses are not used for now. Ones for training & adapting are divided into *{_train,_dev} in dumping while one for evaluation has suffix *_evaluation in dumping
### for training LDA/PLDA: 2 corpora (+1)
sre_tel (cncelebcat_tel) fisher_spa
### for adapting LDA/PLDA: 5 corpora (+1)
sre16_train_dev_cmn sre16_train_dev_ceb sre16_eval_tr60_yue sre16_eval_tr60_tgl sre18_cmn2_train_lab (sre18_dev_unlabeled)
### for evaluation (require: both FE and x-vector extraction): 8 corpora
sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test
#### ***** for sre19_eval_enroll_cmn2 & sre19_eval_test_cmn2, I generate data dir again for them with "_NOdiscard" as their suffices NOT to discard any segments for later steps in back-end

## 1. (ING) x-vector extraction (fisher_spa sre16_eval_tr60_tgl sre16_eval_tr60_yue sre16_train_dev_ceb sre18_cmn2_train_lab sre_tel sre16_train_dev_cmn): This is all labeled data in sre20 recipe except voxcelebcat_tel & swbd. Jesus said swbd is very old phonelines
## < spkid only >
### (DONE) for plda training
for dname in fisher_spa sre16_eval_tr60_tgl sre16_eval_tr60_yue sre16_train_dev_ceb sre18_cmn2_train_lab sre_tel sre16_train_dev_cmn;do
    echo $dname
    bash extract_spkemb_cpuparallel.labcorpus.sh --corpus_name $dname --stage 5 --stop_stage 5 --nj 70 --ngpu 0 --expdir exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup 2>&1 | tee log/spkidonly.bestspkloss.ep50.extraction.sre20.$dname.log &
done
#### (ING) run above again only for sre_tel (this failed due to memory/space error. Check it's log/*sre_tel*error* log if needed) after fixing the "sort -k1" part to "sort -k1 -T tmp_sort/"
### (DONE) do the things for evaluation above (FE and x-vector extraction for the corpora)
#### (DONE at tts_featext) FE
#### (DONE) feature normalization (including copy data from tts_featext to this dir) x-vector extraction
for dname in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test;do
    echo $dname
    bash extract_spkemb_cpuparallel.evalcorpus.sh --corpus_name $dname --stage 0 --stop_stage 5 --nj 70 --ngpu 0 --expdir exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup 2>&1 | tee log/spkidonly.bestspkloss.ep50.extraction.sre20.$dname.log &
done
#### (DONE) (after fix (unbound variable for decode_config), just run stage 5)
for dname in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test;do echo $dname; bash extract_spkemb_cpuparallel.evalcorpus.sh --corpus_name $dname --stage 5 --stop_stage 5 --nj 70 --ngpu 0 --expdir exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup 2>&1 | tee log/spkidonly.bestspkloss.ep50.extraction.sre20.$dname.stage5decodingonly.log & done
##### (ING: Do it for (DONE) spkid, (TODO) spkidtts, (TODO) spkidtts + semi) additional for sre19_eval_enroll_cmn2 & sre19_eval_test_cmn2 with NOdiscard
for dname in sre19_eval_enroll_cmn2_NOdiscard sre19_eval_test_cmn2_NOdiscard;do echo $dname; bash extract_spkemb_cpuparallel.evalcorpus.sh --corpus_name $dname --stage 0 --stop_stage 0 --nj 70 --ngpu 0 --expdir exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup 2>&1 | tee log/spkidonly.bestspkloss.ep50.featnormalization.sre20.$dname.log & done
## < spkid + tts: snapshot.ep.18 >
### (DONE) for plda training
expdir_ori=exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr3e-4_spkloss_weight0.03_fullyunsync_mem35G_ijson_sup # this is long, causing some files to fail to be generated.
expdir=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_unispksampling_lr3e-4_spkloss_weight0.03_mem35G_ijson_sup
mv ${expdir_ori} ${expdir}
for dname in fisher_spa sre16_eval_tr60_tgl sre16_eval_tr60_yue sre16_train_dev_ceb sre18_cmn2_train_lab sre_tel sre16_train_dev_cmn;do
    echo $dname
    bash extract_spkemb_cpuparallel.labcorpus.sh --model snapshot.ep.18 --corpus_name $dname --nj 70 --ngpu 0 --expdir ${expdir} 2>&1 | tee log/spkidtts.manbest_snapshot18.ep30.extraction.sre20.$dname.log &
done
#### (DONE) for evaluation
for dname in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test;do
    echo $dname
    bash extract_spkemb_cpuparallel.evalcorpus.sh --model snapshot.ep.18 --corpus_name $dname --nj 70 --ngpu 0 --expdir ${expdir} 2>&1 | tee log/spkidtts.manbest_snapshot18.ep30.extraction.sre20.$dname.log &
done
## < spkid + tts semi: snapshot.ep.1 >
### (DONE) for plda training
expdir_ori=exp/sre20_train_train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_multicorpora_unispksampling_lr1e-4_transfer_from_supervised_ep14_currentbest_notrainerupdate_spkloss_weight0.03_fullyunsync_mem50G_ijson # this is too long generating errors
expdir=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G
mv ${expdir_ori} ${expdir}
for dname in fisher_spa sre16_eval_tr60_tgl sre16_eval_tr60_yue sre16_train_dev_ceb sre18_cmn2_train_lab sre_tel sre16_train_dev_cmn;do
    echo $dname
    bash extract_spkemb_cpuparallel.labcorpus.sh --model snapshot.ep.1 --corpus_name $dname --nj 70 --ngpu 0 --expdir ${expdir} 2>&1 | tee log/spkidtts_semi.manbest_snapshot1.ep30.extraction.sre20.$dname.log &
done
#### (DONE) for evaluation
for dname in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test;do
    echo $dname
    bash extract_spkemb_cpuparallel.evalcorpus.sh --model snapshot.ep.1 --corpus_name $dname --nj 70 --ngpu 0 --expdir ${expdir} 2>&1 | tee log/spkidtts_semi.manbest_snapshot1.ep30.extraction.sre20.$dname.log &
done





## 2. Average x-vectors (include: 1) WITH sort, cat dump/*{train,dev}/emb* > dump/*trainNdev/emb* && rm dump/*{train,dev}/emb*
## (DONE) < spkid only >
embname=sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup
### (DONE) for training
. path.sh
for corpus_name in sre_tel fisher_spa; do
    bash average_xvector.sh --random true dump/${corpus_name}_{train,dev}/emb.${embname} 2>&1 | tee log/average_xvector.train.${corpus_name}.log
done
### (DONE) for adaptation 
. path.sh
#### (stage 0 DONE (run again after removing NOT completed files): sre16_train_dev_cmn sre16_train_dev_ceb, LEFT (run first time): sre16_eval_tr60_yue sre16_eval_tr60_tgl sre18_cmn2_train_lab)
for corpus_name in sre16_train_dev_cmn sre16_train_dev_ceb sre16_eval_tr60_yue sre16_eval_tr60_tgl sre18_cmn2_train_lab; do
    bash average_xvector.sh dump/${corpus_name}_{train,dev}/emb.${embname} 2>&1 | tee log/average_xvector.adapt.${corpus_name}.log
done
### (DONE) for evaluation
. path.sh
for corpus_name in sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test; do
    bash average_xvector.sh dump/${corpus_name}_evaluation/emb.${embname} 2>&1 | tee log/average_xvector.evaluation.${corpus_name}.log
done
## (DONE) < spkidtts >
embname=sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_unispksampling_lr3e-4_spkloss_weight0.03_mem35G_ijson_sup_snapshot.ep.18
bash local/average_xvector_sre20.sh --embname ${embname} 2>&1 | tee log/average_xvector_sre20.${embname}.log
## (DONE) < spkidtts - semi > From this line, I just put all the lines above for < spkid only > in 2. into local/average_xvector_sre20.sh. With the below commands, simply change the embname for a different xvector extraction model (I ran it for spkid only again by mistake but it should be okay)
embname=sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G_snapshot.ep.1 
bash local/average_xvector_sre20.sh --embname ${embname} 2>&1 | tee log/average_xvector_sre20.${embname}.log

## 2+ Get one xvector per utterance: First, concatenate the segments from the utterance and get one xvector for the utterance
## (DONE) < spkid only >
model_path=exp/sre20_train_train_spkidonly_unsync_shufflebatching_multicorpora_unispksampling_ttslossw0_spkloss_weight1_mem35G_ijson_sup/results/model.loss.best
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
bash local/extract_uttxvector_fromsegs_sre20.sh --nj 70 --model_path ${model_path} 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.log # error after training portion (missed "bash" by mistake)
bash local/extract_uttxvector_fromsegs_sre20_except_trainingset_temp.sh --nj 70 --model_path ${model_path} 2>&1 | tee log/extract_uttxvector_fromsegs_sre20_except_trainingset_temp.${embname}.log # only for adaptation and evaluation
### Do it only for sre19_eval_enroll_cmn2_NOdiscard sre19_eval_test_cmn2_NOdiscard (First time, I think I overwrote to sre19_eval_{enroll,test}_cmn2 without NOdiscard suffix by mistake)
(ING) bash local/extract_uttxvector_fromsegs_sre20.sre19_eval_cmn2_NOdiscard.sh --nj 70 --model_path ${model_path} 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.sre19_eval_cmn2_NOdiscard.${embname}.log
## (ING) < spkidtts >
model_path=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_unispksampling_lr3e-4_spkloss_weight0.03_mem35G_ijson_sup/results/snapshot.ep.18
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
bash local/extract_uttxvector_fromsegs_sre20.sh --nj 70 --model_path ${model_path} 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.log 
## (ING) < spkidtts - semi >
model_path=exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/results/snapshot.ep.1 # dir name was too long to process later parts below so I changed copy it with shorter name
cp -r exp/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G exp/sre20_semi_transferfromsup
model_path=exp/sre20_semi_transferfromsup/results/snapshot.ep.1
embname=`echo ${model_path} | awk -F'exp/' '{print $2}' | awk -F'/' '{print $1"_"$3}'`
bash local/extract_uttxvector_fromsegs_sre20.sh --nj 70 --model_path ${model_path} 2>&1 | tee log/extract_uttxvector_fromsegs_sre20.${embname}.log # stopped after processing until the first corpus in evaluation (File name too long. I changed the names of and some parts in (e.g., by "for cname in ${list_cname};do echo $cname; sed -i 's/sre20_train_ftacotron2+spkemb_2flstm512Nusecat_fullyunsync_shufflebatching_semi_unispksampling_lr1e-4_transfer_fromsup_ep14_currentbest_notrainerupdate_spkloss_weight0.03_mem50G/sre20_semi_transferfromsup/g' dump/${cname}_trainNdev/emb_fromcatsegs_sre20_semi_transferfromsup_snapshot.ep.1/xvector.scp; head -n1 dump/${cname}_trainNdev/emb_fromcatsegs_sre20_semi_transferfromsup_snapshot.ep.1/xvector.scp; done") all the generated files by the time as above with shorter name and ran one below).
bash local/extract_uttxvector_fromsegs_sre20_onlyevalcorpora.sh --nj 70 --model_path ${model_path} 2>&1 | tee log/extract_uttxvector_fromsegs_sre20_onlyevalcorpora.${embname}.log # only run for evaluation corpora

## 3. (RUN THIS at /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1) Combine data (similar to the process left after x-vector extraction in run_030_extract_xvectors.sh under /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1/) and run backend training and evaluation
## (DONE) < spkid only >: Refer to /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1/list_run.sh
## (ING) < spkidtts >
## (ING) < spkidtts - semi >

## 4. (RUN THIS at /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1) backend training and evaluation
## (ING) < spkid only >: Refer to /export/b16/jcho/hyperion_20210108/egs/sre20-cts/v1/list_run.sh




# debug
bash run.asrttsspkid_ijson.debug.sh --gpu_ix $(free-gpu) --stage 4 --stop_stage 4
## memory profile (check one of the below files)
run.asrttsspkid.memoryprofile.debug.sh
run.asrttsspkid_ijson.unispksampling.mem30G.memoryprofile.sh # e.g., bash run.asrttsspkid_ijson.unispksampling.mem30G.memoryprofile.sh --stage 4 --stop_stage 4 --gpu_ix $(free-gpu) OR change memoryprofile to debug if you want debugging
# run.asrttsspkid_ijson.unispksampling.removetextNtoken.mem30G.debug.sh includes removing 'text' and 'token' in json dictionaries to reduce memory usage
