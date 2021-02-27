. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

expdir=exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_noatt_rf3_spkloss_weight0/

. utils/parse_options.sh || exit 1;

logname=`basename ${expdir}`
for ix in {1..30};do
    bash run_backendonly_cpuparallel.sh --model snapshot.ep.${ix} --nj 70 --ngpu 0 --expdir ${expdir} 2>&1 | tee log/backend.${logname}.snapshot.ep.${ix}.log
done
