for fname in `ls -d exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight*`;do
    (
    logname=$(basename ${fname})
    echo ${logname}
    bash run_backendonly_cpuparallel.sh --nj 1 --ngpu 0 --expdir ${fname} 2>&1 | tee log/backend.${logname}.log # not parallel running since I do not have much slots available currently
    ) &
done
