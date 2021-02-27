# ********** fbank features here are extracted using ESPnet make_fbank.sh with
# stft's center=False (that correspondings to --snip-edges=true in kaldi
# version) ********* BUT will use grifinlim with ceneter=True. This
# inconsistency did not give any problem in one sample experiment

# before run, the output of "$ls -ltrd data/*"
************************************************************************
drwxr-xr-x 2 jcho fax 4096 Apr  7 18:17 data/test_clean
drwxr-xr-x 2 jcho fax 4096 Apr  7 18:20 data/train_clean_100
drwxr-xr-x 2 jcho fax 4096 Apr  7 18:30 data/train_clean_360
drwxr-xr-x 5 jcho fax 4096 Apr  7 18:30 data/dev_clean_org
drwxr-xr-x 4 jcho fax 4096 Apr  7 18:30 data/dev_clean
drwxr-xr-x 4 jcho fax 4096 Apr  7 23:02 data/train_clean_460
drwxr-xr-x 5 jcho fax 4096 Apr  7 23:02 data/train_clean_460_org
************************************************************************
# voxceleb1
bash run.FE.diffconfig.voxceleb1.sh --stage 0 --stop_stage 1 2>&1 | tee log/run.FE.diffconfig.voxceleb1.fromstage0to1.log

# voxceleb1 FE left parts after filtered voxceleb1 FE above. This FE is to generate all voxceleb1 trial lists
## start
. path.sh
cp -r /export/b13/jcho/espnet_v3/tools/kaldi/egs/voxceleb/v2/data/voxceleb1 data/
### 1. get diff utt list bet. the original and filtered ones
mkdir -p data/voxceleb1_notfiltered
filter_scp.pl --exclude <(cat data/voxceleb1_train_filtered/utt2spk data/voxceleb1_test/utt2spk) data/voxceleb1/utt2spk > data/voxceleb1_notfiltered/utt2spk
### 2. copy other files from voxceleb1 to voxceleb1_notfiltered & fix the directory
cp `ls data/voxceleb1/* | grep -v utt2spk` data/voxceleb1_notfiltered/
fix_data_dir.sh data/voxceleb1_notfiltered/
### 3. feature extraction with librosa
bash run.FE.diffconfig.voxceleb1_notfiltered.sh 2>&1 | tee log/run.FE.diffconfig.voxceleb1_notfiltered.log
### 4. combine dir
rm -rf data/voxceleb1 && utils/combine_data.sh data/voxceleb1 data/voxceleb1_train_filtered data/voxceleb1_test data/voxceleb1_notfiltered
## end


# (working) adding pitch
bash run.FE.diffconfig.voxceleb1.pitchadd.sh 2>&1 | tee log/run.FE.diffconfig.voxceleb1.pitchadd.log
