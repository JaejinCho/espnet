# copy-feats for original features (normalized), synthesized features (normalized & denormalized)
# 1. in libritts train_dev subset (for asrtts+spkid w/ spkloss_weight0.001)
. path.sh
copy-feats scp:dump/train_dev/feats.scp ark,t:ori.feats.norm.libritts.txt
copy-feats scp:exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.001/outputs_model.loss.best_decode/train_dev/feats.scp ark,t:syn.norm.libritts.txt
copy-feats scp:exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.001/outputs_model.loss.best_decode_denorm/train_dev/feats.scp ark,t:syn.denorm.libritts.txt

# 2. in voxceleb1 voxceleb1_test subset (for asrtts+spkid w/ spkloss_weight0)
copy-feats scp:dump/voxceleb1_train_filtered_dev/feats.scp ark,t:ori.feats.norm.voxceleb1.txt
copy-feats scp:exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0/outputs_model.loss.best_decode/voxceleb1_train_filtered_dev/feats.scp ark,t:syn.norm.voxceleb1.txt
copy-feats scp:exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0/outputs_model.loss.best_decode_denorm/voxceleb1_train_filtered_dev/feats.fbankonly.scp ark,t:syn.denorm.voxceleb1.txt
copy-feats scp:exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0/outputs_model.loss.best_decode_denorm_16kVoceleb1Config/voxceleb1_train_filtered_dev/feats.fbankonly.scp ark,t:syn.denorm_16kVoceleb1Config.voxceleb1.txt
