eval_set=train_eval # for now, it is fixed all the time

for fname in `ls -d exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight*| grep -wv "train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0$\|train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0_old_WOspkidacc$\|train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight1$\|train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight3$\|train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight10$"`;do
    embname=$(basename ${fname})
    qsub -cwd -o log/qsub.run.eer_mindcf.${embname}.log -e log/qsub.run.eer_mindcf.${embname}.log -l mem_free=200G,ram_free=200G run.eer_mindcf.sh --embname ${embname}
done
