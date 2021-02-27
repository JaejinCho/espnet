# ********** fbank features here are extracted using ESPnet make_fbank.sh with
# stft's center=False (that correspondings to --snip-edges=true in kaldi
# version) ********* BUT will use grifinlim with ceneter=True. This
# inconsistency did not give any problem in one sample experiment

# before run, the output of "$ls -ltrd data/*"
drwxr-xr-x 2 jcho fax 4096 Apr  7 18:17 data/test_clean
drwxr-xr-x 2 jcho fax 4096 Apr  7 18:20 data/train_clean_100
drwxr-xr-x 2 jcho fax 4096 Apr  7 18:30 data/train_clean_360
drwxr-xr-x 5 jcho fax 4096 Apr  7 18:30 data/dev_clean_org
drwxr-xr-x 4 jcho fax 4096 Apr  7 18:30 data/dev_clean
drwxr-xr-x 4 jcho fax 4096 Apr  7 23:02 data/train_clean_460
drwxr-xr-x 5 jcho fax 4096 Apr  7 23:02 data/train_clean_460_org
drwxr-xr-x 2 jcho fax   46 Apr 13 00:03 data/voxceleb1_train_filtered
drwxr-xr-x 2 jcho fax  128 Apr 13 00:16 data/voxceleb1_test
drwxr-xr-x 5 jcho fax 4096 Apr 13 00:18 data/voxceleb1_train_filtered_dev_org
drwxr-xr-x 4 jcho fax 4096 Apr 13 00:18 data/voxceleb1_train_filtered_dev
drwxr-xr-x 5 jcho fax 4096 Apr 13 00:26 data/voxceleb1_train_filtered_train_org
drwxr-xr-x 4 jcho fax 4096 Apr 13 00:30 data/voxceleb1_train_filtered_train
#
(DONE) bash run.FE.diffconfig.voxceleb_kaldiprep.sh --stop_stage 1 2>&1 | tee log/run.FE.diffconfig.voxceleb_kaldiprep.untilstage1.log
(DONE) bash run.FE.diffconfig.voxceleb_kaldiprep.sh --stage 2 --stop_stage 3 2>&1 | tee log/run.FE.diffconfig.voxceleb_kaldiprep.stage2N3.log # this was run again after changing kaldi link up-to-date ($ rm /export/b18/jcho/espnet3/tools/kaldi && ln -s /export/b11/jcho/kaldi_20190316 /export/b18/jcho/espnet3/tools/kaldi)
(RELATED FILES ARE ALL REMOVED. THIS IS DONE IN /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxcelebNaug_kaldiasr_phonealign_rf3/run.asrttsspkid.spkloss_weight.new.update.rf3.sh) bash run.FE.diffconfig.voxceleb_kaldiprep.sh --stage 4 --stop_stage 4 2>&1 | tee log/run.FE.diffconfig.voxceleb_kaldiprep.stage4only.log
## Later parts including the one above will be done in a new directory (e.g.) dump with normalization, train, etc.)
