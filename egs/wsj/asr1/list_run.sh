# Decoding libritts train_train and train_dev (using a model trained with wsj corpus)
bash run_decoding.libritts.sh --ngpu 1 2>&1 | tee log/run_decoding.libritts.log

# Multi-gpu training (Currently targetting to use 2 gpus)
(DONE, Data prep) bash run_attonly_2gpu.sh --ngpu 2 --stop_stage 4 2>&1 | tee log/run_attonly_2gpu.log
## below 2 are to compare speed between using 1 gpu and 2 gpus
(ING) bash run_attonly_2gpu.sh --ngpu 2 --stage 4 --stop_stage 4 2>&1 | tee log/run_attonly_2gpu.stage4only.log
(ING) bash run_attonly_1gpu.sh --ngpu 1 --stage 4 --stop_stage 4 2>&1 | tee log/run_attonly_1gpu.stage4only.log
