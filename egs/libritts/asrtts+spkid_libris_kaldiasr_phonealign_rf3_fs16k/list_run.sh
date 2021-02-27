mkdir log

# 1. Training
(STOPPED at stage 4 to run below: fullyunsync) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fs16k.sh --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fs16k.spkloss_weight0.log
(ING) bash run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.fs16k.sh --stage 4 --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.update.rf3.fullyunsync.fs16k.spkloss_weight0.fromstage4.log
