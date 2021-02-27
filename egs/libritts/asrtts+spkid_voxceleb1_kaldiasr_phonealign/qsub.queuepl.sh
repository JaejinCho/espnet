. path.sh
. cmd.sh
data2json_kaldiali.sh --cmd "${train_cmd}" --nj 70 --kaldiali true --feat dump/voxceleb1_train_filtered_train/feats.scp data/voxceleb1_train_filtered_train data/lang_1char/voxceleb1_train_filtered_train_units.txt > data.tmp.nj70.json
