# *** NOfull-length speaker embedding experiments were done ***
# - Directory description: This directory is edited from
# ../../../egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3
# to add pitch as a decoder input
# - What's changed:
## 1. data prep
## 2. espnet codes (newly created below, roughly top to bottom. They all include "addpitch" in the names):
### 1) run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.sh
### 2) ../../../espnet/bin/tts_train_speakerid_addpitch.py
### 3) ../../../espnet/tts/pytorch_backend/tts_speakerid_addpitch.py
### 4) ../../../espnet/nets/pytorch_backend/e2e_tts_tacotron2_speakerid_update_unsync_addpitch.py
### 5) ../../../espnet/nets/pytorch_backend/tacotron2/decoder_update_addpitch.py
### 6) ../../../espnet/utils/io_utils_speakerid_addpitch.py


# *** Actuall code run ***
mkdir -p log
# 1. Train
## debugging to fix parts of the code
qlogin -l mem_free=20G -l gpu=1 -l "hostname=b1[12345678]*|c0[1345678]*|c1[01]*" -q i.q -now no
bash run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.debug.stage4.sh --gpu_ix $(free-gpu) --stage 4 --stop_stage 4 # spkloss_weight=0
bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.debug.stage4.sh --gpu_ix $(free-gpu) --stage 4 --stop_stage 4 # spkloss_weight=0
## Actually running after finishing coding
(*** only for the first run ***) bash run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.sh --ngpu 1 --n_average 0 --spkloss_weight 0.03 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.spkloss_weight0.03.log
(DONE) bash run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.sh --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.spkloss_weight0.03.onlystage4.log
(DONE) bash run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.sh --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.spkloss_weight0.onlystage4.log
## (+ specaugtts)
cp /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/conf/specaug.yaml conf/
(DONE) bash run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.specaugtts.sh --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.specaugtts.spkloss_weight0.03.onlystage4.log
(DONE, run from ep.30 to ep.60) bash run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.specaugtts.sh --train_config conf/train_pytorch_tacotron2+spkemb_noatt_rf3_epoch60.yaml --resume exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_specaug_spkloss_weight0.03_unsync_addpitch/results/snapshot.ep.30 --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.specaugtts.spkloss_weight0.03.onlystage4.from30to60epochs.from_modellossbest_ep.30.log
(DONE) bash run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.specaugtts.sh --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.specaugtts.spkloss_weight0.onlystage4.log
(DONE, run from ep.30 to ep.60) bash run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.specaugtts.sh --train_config conf/train_pytorch_tacotron2+spkemb_noatt_rf3_epoch60.yaml --resume exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_specaug_spkloss_weight0_unsync_addpitch/results/snapshot.ep.30 --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.unsync.addpitch.specaugtts.spkloss_weight0.onlystage4.from30to60epochs.from_modellossbest_ep.30.log

## fully unsync. *** Refer to coding_progress.txt for how I edited codes
### Newly generated codes (most of them are edited from some codes):
### - run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.sh
(STOPPED, 5Gram below was fine to run) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.sh --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.spkloss_weight0.03.onlystage4.log
(STOPPED, 5Gran below was fine to run) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.sh --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.spkloss_weight0.onlystage4.log
# with small RAM request
(DONE) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.5Gram.sh --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.spkloss_weight0.03.5Gram.onlystage4.log
(DONE) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.5Gram.sh --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.addpitch.spkloss_weight0.5Gram.onlystage4.log


# 2. Backend
## spkid_loss weight 0.03
### unsync
(DONE (EER 3.40), best loss, ep.28) # "validation/main/loss": 0.8063927958237713, "validation/main/spkid_loss": 0.41625848916712505, "validation/main/spkid_acc": 0.9900900554187191
bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_unsync_addpitch/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_unsync_addpitch.log
(DONE (EER 3.38), best spkid_loss, ep.30) #"validation/main/loss": 0.8081737576887525, "validation/main/spkid_loss": 0.387859433688673, "validation/main/spkid_acc": 0.9898848009031198
bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.30 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_unsync_addpitch/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_unsync_addpitch.snapshot.ep.30.spkid_lossbest.log
(DONE (EER 3.56), best spkid_acc, ep.25) # "validation/main/loss": 0.8106251132899317, "validation/main/spkid_loss": 0.4339988803037378, validation/main/spkid_acc": 0.990288895730706
bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.25 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_unsync_addpitch/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_unsync_addpitch.snapshot.ep.25.spkid_accbest.log
(DONE, +specaug, best loss, ep.30) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_specaug_spkloss_weight0.03_unsync_addpitch/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_specaug_spkloss_weight0.03_unsync_addpitch.log
(ING, +specaug, best loss, ep.56) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_epoch60_specaug_spkloss_weight0.03_unsync_addpitch/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_specaug_spkloss_weight0.03_unsync_addpitch.bestloss.epoch56of60.log
### fullysync
(DONE, best loss) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_fullyunsync_addpitch_5Gram/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_fullyunsync_addpitch_5Gram.log
## spkid_loss weight 0
### unsync
(DONE) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_unsync_addpitch/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_unsync_addpitch.log
(DONE, +specaug, best loss, ep.30) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_specaug_spkloss_weight0_unsync_addpitch/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_specaug_spkloss_weight0_unsync_addpitch.log
(ING, +specaug, best loss, ep.43) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_epoch60_specaug_spkloss_weight0_unsync_addpitch/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_epoch60_specaug_spkloss_weight0_unsync_addpitch.bestloss.epoch43of60.log
### fullysync (It seems training over more epochs lead to better EER until 30 epochs at least)
(DONE (EER 5.74), best loss & mse_loss, ep.28) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_fullyunsync_addpitch_5Gram/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_fullyunsync_addpitch_5Gram.log
(DONE (EER 5.95), best l1_loss, ep.26) bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.26 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_fullyunsync_addpitch_5Gram/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_fullyunsync_addpitch_5Gram.snapshot.ep.26.l1_lossbest.log
(DONE (EER 5.63), just the last epoch, ep.30) bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.30 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_fullyunsync_addpitch_5Gram/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0_fullyunsync_addpitch_5Gram.snapshot.ep.30.log
