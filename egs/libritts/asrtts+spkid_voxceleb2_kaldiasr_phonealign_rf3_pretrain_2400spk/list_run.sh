mkdir log

# 1. Training
(*** only for the first run ***) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.pretrain.sh --train_set voxceleb2_2400spk --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.spkloss_weight0.voxceleb2_2400spk.pretrain.log
