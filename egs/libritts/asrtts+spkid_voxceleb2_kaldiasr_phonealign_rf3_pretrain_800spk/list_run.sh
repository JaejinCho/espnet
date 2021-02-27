mkdir log

# 1. Training
(*** only for the first run ***) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.pretrain.sh --train_set voxceleb2_800spk --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.spkloss_weight0.voxceleb2_800spk.pretrain.log



# 2. Backend
(ING) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_fullyunsync/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0.03_fullyunsync.log
