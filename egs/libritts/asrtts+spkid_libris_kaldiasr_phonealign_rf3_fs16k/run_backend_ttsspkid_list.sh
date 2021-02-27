for fname in `ls -d exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight*| grep -wv "train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0$\|train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0_old_WOspkidacc$\|train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight1$\|train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight3$\|train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight10$"`;do
    (
    logname=$(basename ${fname})
    echo ${logname}
    bash run_backendonly_cpuparallel.sh --nj 1 --ngpu 0 --expdir ${fname} 2>&1 | tee log/backend.${logname}.log # not parallel running since I do not have much slots available currently
    ) &
done
