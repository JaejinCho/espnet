# run tts for libritts (Training)
bash run.sh --datadir /export/a15/jcho/data 2>&1 | tee log/run.datadir.log
bash run.sh --stage 4 --datadir /export/a15/jcho/data 2>&1 | tee log/run.datadir.stage4.log # after changing to "cmd_backend='jhu'" in cmd.sh
(Fastspeech, placeholder teacher model) bash run.sh --stage 4 --train_config conf/train_pytorch_tacotron2+spkemb_ep1_rf1.yaml 2>&1 | tee log/run.teacher.placeholder.log
(Fastspeech) bash run.sh --stage 4 --train_config conf/train_pytorch_fastspeech.yaml 2>&1 | tee log/run.fastspeech.stage4.log

# For debug
bash run.debug.sh --stage 4 --gpu_ix $(free-gpu)
(Fastspeech) bash run.debug.sh --stage 4 --train_config conf/train_pytorch_fastspeech.yaml --gpu_ix $(free-gpu)
(FFtts) bash run.debug.sh --stage 4 --train_config conf/train_pytorch_fastspeech_speakerid.yaml --gpu_ix $(free-gpu)

# To generated speaker id added (instead of xvector features) data.json
bash update_json.temp.sh 2>&1 | tee log/update_json.spkid.temp.sh

# (+) additional feature extraction with kaldi bin to match # input frames and # output frames in kaldi phoneme alignments TTS training
## (practice or small check) -> it turns out for TTS I have to use make_fbank.sh in ESPnet for feature extraction for consistency
steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
    data/${subset} exp/make_fbank/${subset} ${fbankdir}
utils/fix_data_dir.sh data/${subset}
## feature extraction with different configuration

