mkdir log

# 1. Data prep: spkloss_weight does not matter specified here in data prep.
(DONE, *** only for the first run ***) bash run.asrttsspkid.sh --stage 0 --stop_stage 2 --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/dataprep.stagefrom0to2.log

cp /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_nceloss_fullyunsync_shufflebatching_bs56.yaml conf/
# 2. Training (2flstm512Nusecat + shufflebathcing (fixed bs) + fullyunsync with spkloss_weight=0.03 and nceloss=0.001)
(ING) bash run.asrttsspkid.sh --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_nceloss_fullyunsync_shufflebatching_bs56.yaml --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 --nceloss_weight 0.001 2>&1 | tee log/run.asrttsspkid.spkloss_weight0.03.nceloss_weight0.001.onlystage4.2flstm512Nusecat.fullyunsync.shufflebatching.bs56.log

# -1. Debug
qlogin -l mem_free=20G -l gpu=1 -l "hostname=b1[12345678]*|c0[1345678]*|c1[01]*" -q i.q -now no
## ipdb at break point
bash run.asrttsspkid.debug.stage4.sh --gpu_ix $(free-gpu) --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_nceloss_fullyunsync_shufflebatching_bs56.yaml --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 --nceloss_weight 0.001
## ipdb stops at an error. You have to enter "c" in the beginning
bash run.asrttsspkid.newdebug.stage4.sh --gpu_ix $(free-gpu) --train_config conf/train_pytorch_forwardtacotron2+spkemb_2flstm512Nusecat_nceloss_fullyunsync_shufflebatching_bs56.yaml --stage 4 --stop_stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0.03 --nceloss_weight 0.001

