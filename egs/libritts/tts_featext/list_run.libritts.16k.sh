# feature extraction
bash run.FE.diffconfig.libritts.16k.sh --stage -1 --stop_stage 1 2>&1 | tee log/run.FE.diffconfig.libritts.16k.fromstage_1to1.log

# copy required text files
cp data/train_clean_460/{train_train.uttlist,train_dev.uttlist,train_eval.uttlist,train_trainNdev.utt2spklab,train_trainNdev.utt2spklab_NOspk2085} data/train_clean_460_16k/
