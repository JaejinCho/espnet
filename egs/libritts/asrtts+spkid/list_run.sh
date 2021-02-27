mkdir -p log

## Libritts TTS training with wsj ASR transcripts

# 1. Training
# tts + spkid joint (*** Run first one below first for data prep. Then, later experiments make sure to run from stage 4 (in most cases) ***)
bash run.asrttsspkid.spkloss_weight.new.sh --ngpu 1 --n_average 0 --spkloss_weight 0.003 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.spkloss_weight0.003.log
bash run.asrttsspkid.spkloss_weight.new.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0.01 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.spkloss_weight0.01.log
bash run.asrttsspkid.spkloss_weight.new.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0.03 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.spkloss_weight0.03.log
bash run.asrttsspkid.spkloss_weight.new.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0.3 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.spkloss_weight0.3.log
bash run.asrttsspkid.spkloss_weight.new.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 3 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.spkloss_weight3.log
bash run.asrttsspkid.spkloss_weight.new.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.spkloss_weight0.log

## with Xformer ASR transcription as input
(ONLY FOR THE FIRST RUN) bash run.asrttsspkid.spkloss_weight.new.librispeechXformer.sh --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.librispeechXformer.spkloss_weight0.log
bash run.asrttsspkid.spkloss_weight.new.librispeechXformer.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.new.librispeechXformer.spkloss_weight0.fromstage4.log

# pretrained xvector
bash run.asrtranscript.pretrained_xvector.sh --ngpu 1 --n_average 0 2>&1 | tee log/run.asrtranscript.pretrained_xvector.log

# 2. Backend evaluation for SV
bash run_backend_ttsspkid_list.sh # it does NOT incude the run for pretrained_xvector
(REDO 1: files that has errors as "run_backendonly_cpuparallel.sh: line 142: queue.pl: command not found" at stage 6)
bash run_backendonly_cpuparallel.sh --stage 6 --nj 1 --ngpu 0 --expdir exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.01 2>&1 | tee log/backend.train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.01.stage6.log
bash run_backendonly_cpuparallel.sh --stage 6 --nj 1 --ngpu 0 --expdir exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.03 2>&1 | tee log/backend.train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.03.stage6.log
bash run_backendonly_cpuparallel.sh --stage 6 --nj 1 --ngpu 0 --expdir exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.3 2>&1 | tee log/backend.train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.3.stage6.log
bash run_backendonly_cpuparallel.sh --stage 6 --nj 1 --ngpu 0 --expdir exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight3 2>&1 | tee log/backend.train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight3.stage6.log
(REDO 2: 1) weight0 has some error I do NOT know why, 2) 0.001 ran twice at the same time)
(Stopped since the jobs sent to a node (for train_train) was so slow) bash run_backendonly_cpuparallel.rerun.sh --nj 1 --ngpu 0 --expdir exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0 2>&1 | tee log/backend.train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.rerun.log
bash run_backendonly_cpuparallel.rerun.sh --nj 1 --ngpu 0 --expdir exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.001 2>&1 | tee log/backend.train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.001.rerun.log
(REDO 3: Rerun the stopped job at REDO 2 with more nj. Have run below in order)
bash run_backendonly_cpuparallel.rerun.onlytrain_train.sh --stop_stage 5 --nj 64 --ngpu 0 --expdir exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0 2>&1 | tee log/backend.train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.rerun.onlytrain_train.nj64.log
bash run_backendonly_cpuparallel.rerun.onlytrain_train.sh --stage 6 --ngpu 0 --expdir exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0 2>&1 | tee log/backend.train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.rerun.onlytrain_train.nj64.fromstage6.log

(Run list for backend training)
exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0
exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.001
exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.01
exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.03
exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight0.3
exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_spkloss_weight3

# backend
(have run but run the below again to make sure) for w in 0.001 0.01 0.03 0.3 3;do echo $w; bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight${w} 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight${w}.lossbest.log; done
(ING) for w in 0.001 0.01 0.3 3;do echo $w; bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight${w} 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight${w}.lossbest.log; done





##### With filtered data (144723 filtered from 148642 utts with the config "--maxframes 3000 --maxchars 400". In the end 138955 for training and 5768 for model selection (dev))
# 1. Voxceleb1 TTS training with wsj ASR transcripts (NO human transcript labeled so the directory has text is just text.asr (in Libritts case))
## 1) Preprocessing
### filter voxceleb_train to voxceleb_train_filtered
. path.sh
remove_longshortdata.sh --maxframes 3000 --maxchars 400 /export/b11/jcho/espnet/egs/wsj/asr1/data/voxceleb1_train /export/b11/jcho/espnet/egs/wsj/asr1/data/voxceleb1_train_filtered
### split the voxceleb1_train data for TTS training (speaker overlap between train and val sets)
python make_uttlists_voxceleb1_filtered.py

## 2) run training script (Currently, do not do things for eval_set. To include this, run feature extraction first for eval_set (in /export/b11/jcho/espnet/egs/wsj/asr1/run_decoding.voxceleb1.cpu.voxceleb1_train.sh) and uncomment the related lines in the script below)
bash run.asrttsspkid.spkloss_weight.voxceleb1_filtered.sh --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.log # first run
bash run.asrttsspkid.spkloss_weight.voxceleb1_filtered.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.fromstage4.log # from second run (once first run data prep part run okay)
bash run.asrttsspkid.spkloss_weight.voxceleb1_filtered.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight [spkloss_weight] 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.fromstage4.log # spkloss_weight for 0.001, 0.01, 0.03, 0.3, 3
bash run.asrttsspkid.spkloss_weight.voxceleb1_filtered_bs48.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.bs48.fromstage4.log # spkloss_weight for 0, 0.001, 0.01, 0.03, 0.3, 3 # ******* bs48 due to memory error *******

## 3) Synthesize only
bash run.asrttsspkid.spkloss_weight.voxceleb1_filtered_bs48.synthesizeonly.sh --ngpu 1 --stage 6 --stop_stage 6 --n_average 0 --spkloss_weight [value] --nj 32 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.bs48.synthesizeonly.log # spkloss_weight # for 0, 0.03
(FINAL FIX ACCORDING TO THE ORIGINAL FEATURES CONFIGURATION) bash run.asrttsspkid.spkloss_weight.voxceleb1_filtered_bs48.synthesizeonly.16k.sh --ngpu 1 --stage 6 --stop_stage 6 --n_average 0 --spkloss_weight 0 --fs 16000 --n_fft 600 --n_shift 160 --nj 32 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.bs48.synthesizeonly.16k.log # set --fs 16000 --n_fft 600 --n_shift 160 (window_length: 25 msec, hop_size: 10 msec at sampling frequency 16k). Run for both 0 and 0.03

## (+) consistency loss
(NOT WORKING) bash run.asrttsspkid.spkloss_weight.consloss_weight.voxceleb1_filtered_bs48.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0 --consloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.consloss_weight0.bs48.fromstage4.log # spkloss_weight for 0 + consloss_weihgt{0,0.01,0.03,0.1,1,2,3} ******* bs48 due to memory error *******
(STOPPED DUE TO GRID ERRORS) bash run.asrttsspkid.spkloss_weight.consloss_weight.voxceleb1_filtered_bs36.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0 --consloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.consloss_weight0.bs36.fromstage4.log # spkloss_weight for 0 + consloss_weihgt{0,0.01,0.03,0.1,1,2,3} ******* bs36 due to memory error *******
(RUNNING) bash run.asrttsspkid.spkloss_weight.consloss_weight.voxceleb1_filtered_bs36.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0 --consloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.spkloss_weight0.consloss_weight0.bs36.fromstage4.log # spkloss_weight for 0 + consloss_weiggt{0,0.01,0.03,0.1,1,10} ******* bs36 due to memory error *******: (DONE: 0.01 // RERUN FROM STOP (DUE TO MEMORY ERROR <= debug need at some point): 0, 0.1, 1, 10 // STOPPED: // RUNNING: 0.03)
(RUNNING) bash run.asrttsspkid.spkloss_weight.consloss_weight.voxceleb1_filtered.detach.sh --ngpu 1 --batch-size 40 --stage 4 --n_average 0 --spkloss_weight 0 --consloss_weight 1 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.detach.spkloss_weight0.consloss_weight1.bs40.fromstage4.log ******* bs40 due to memory error *******: (RUNNING: 1, WILL RUN: 0.01, 0.1, 10, 100)

# spkid only (0. bs64 rather than bs48 differently of TTS + spkid cases 1. DID NOT ADD "filtered" after "voxceleb1" in naming the run script and the log. Change them after the experiment is done)
## with whole utt
bash run.spkidonly.final.voxceleb1.sh --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.spkidonly.final.voxceleb1.stage4only.log
## with chunks
bash run.spkidonly.final.voxceleb1.chunks.sh --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.spkidonly.final.voxceleb1.chunks.from200to400.stage4only.log
bash run.spkidonly.final.voxceleb1.fixedchunks.sh --chunk_len 200 --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.spkidonly.final.voxceleb1.fixedchunks.200.stage4only.log
bash run.spkidonly.final.voxceleb1.fixedchunks.sh --chunk_len 400 --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.spkidonly.final.voxceleb1.fixedchunks.400.stage4only.log
(DONE) bash run.spkidonly.final.voxceleb1.chunks.shufflebatching.sh --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.spkidonly.final.voxceleb1.chunks.shufflebatching.bs128.from200to400.stage4only.log
(DONE) bash run_fbankonly.spkidonly.final.voxceleb1.chunks.shufflebatching.sh --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --n_average 0 --stop_stage 4 2>&1 | tee log/run_fbankonly.spkidonly.final.voxceleb1.chunks.shufflebatching.bs128.from200to400.log
(ING, fbankonly + longer window) bash run_fbankonly.spkidonly.final.voxceleb1.chunks.shufflebatching.sh --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --n_average 0 --stop_stage 4 2>&1 | tee log/run_fbankonly.spkidonly.final.voxceleb1.chunks.shufflebatching.bs128.from200to400.log
## specaug by Following the specaug parameters in the paper, INVESTIGATION OF SPECAUGMENT FOR DEEP SPEAKER EMBEDDING LEARNING
(ING, with specaug_rate=1.0) bash run.spkidonly.final.voxceleb1.chunks.shufflebatching.specaug.sh --specaug_rate 1.0 --train_config conf/train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_specaug_bs128.yaml --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.spkidonly.final.voxceleb1.chunks.shufflebatching.bs128.from200to400.stage4only.specaug_rate1.folowingpaper.log
(ING, with specaug_rate=0.5) bash run.spkidonly.final.voxceleb1.chunks.shufflebatching.specaug.sh --specaug_rate 0.5 --train_config conf/train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_specaug_bs128.yaml --ngpu 1 --ttsloss_weight 0 --spkloss_weight 1 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.spkidonly.final.voxceleb1.chunks.shufflebatching.bs128.from200to400.stage4only.specaug_rate0.5.followingpaper.log

# 2. Voxceleb1 TTS training with librispeechXformer ASR transcripts (NO human transcript labeled so the directory has text is just text.asr (in Libritts case))
## 1) Preprocessing (*** RUN THIS AT /export/b18/jcho/espnet_asr_transformer_final/egs/librispeech/asr1 ***)
# generate text.asr
python rec_text2text.py exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/decode_voxceleb1_train_datadist_C_model.val5.avg.best_decode_pytorch_transformer_large.beamsize10_pretrained_cpudecoding/data.json data/voxceleb1_train/
# change text.asr to text
mv data/voxceleb1_train/text data/voxceleb1_train/.text && mv data/voxceleb1_train/text.asr data/voxceleb1_train/text
# copy the filtered directory from the wsj asr config & change the text file to librittsXformer asr transcription
cp -r /export/b11/jcho/espnet/egs/wsj/asr1/data/voxceleb1_train_filtered/ data/ && cp data/voxceleb1_train/text data/voxceleb1_train_filtered/

## 2) run training script
bash run.asrttsspkid.spkloss_weight.voxceleb1_filtered_bs48.librispeechXformer.sh --ngpu 1 --stage [0 if the first run, 4 after the data is prepared ahead] --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1_filtered.librispeechXformer.spkloss_weight0.bs48.fromstage4.log # spkloss_weight for 0, 0.001, 0.01, 0.03, 0.3, 3 # ******* bs48 due to memory error *******

## (+) consistency loss
(RUNNING) bash run.asrttsspkid.spkloss_weight.consloss_weight.voxceleb1_filtered.librispeechXformer.detach.sh --ngpu 1 --batch-size 40 --stage 4 --n_average 0 --spkloss_weight 0 --consloss_weight 1 2>&1 | tee log/run.asrttsspkid.spkloss_weight.consloss_weight.voxceleb1_filtered.librispeechXformer.detach.spkloss_weight0.consloss_weight1.bs40.fromstage4.log ******* bs40 due to memory error *******: (RUNNING: 1, WILL RUN: 0.01, 0.1, 10, 100)

###### (WILL RUN IF NEEDED) With all data (148642 utts) - may need to cut into pieces the utterances longer than 3000 frames
## Voxceleb1 TTS training with wsj ASR transcripts (NO human transcript labeled so the directory has text is just text.asr (in Libritts case))
# split the voxceleb1_train data for TTS training (speaker overlap between train and val sets)
python make_uttlists_voxceleb1.py
# run training script (Currently, do not do things for eval_set. To include this, run feature extraction first for eval_set (in /export/b11/jcho/espnet/egs/wsj/asr1/run_decoding.voxceleb1.cpu.voxceleb1_train.sh) and uncomment the related lines in the script below)
bash run.asrttsspkid.spkloss_weight.voxceleb1.sh --ngpu 1 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1.spkloss_weight0.log # first run
bash run.asrttsspkid.spkloss_weight.voxceleb1.sh --ngpu 1 --stage 2 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1.spkloss_weight0.fromstage2.log # error due to eval_set so run from stage 2 again
bash run.asrttsspkid.spkloss_weight.voxceleb1.sh --ngpu 1 --stage 4 --n_average 0 --spkloss_weight 0 2>&1 | tee log/run.asrttsspkid.spkloss_weight.voxceleb1.spkloss_weight0.fromstage4.log # from second run (once first run data prep part run okay)




# Backend evaluation for SV
## cp voxceleb1_test directory
cp -r /export/b11/jcho/espnet/egs/wsj/asr1/data/voxceleb1_test data/
(*** WRONG NORMALIZAITON. FOUND VERY LATE) cp -r /export/b11/jcho/espnet/egs/wsj/asr1/dump/voxceleb1_test/deltafalse dump/voxceleb1_test # To be consistent with directory structures of the other subsets in dump/ directory (They do not have deltafalse subdirectory. I think this is due to difference between TTS and ASR recipes)
(*** CORRECTED NORMALIZATION) mv dump/voxceleb1_test dump/voxceleb1_test_wrongnorm && bash run.dump_voxceleb1_test.sh 2>&1 | tee log/run.dump_voxceleb1_test.log
## (*** Wrong SV eval due to the wrong normalization above*** ) run the backend exps - start
bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.log
bash run_backendonly_cpuparallel.voxceleb1.sh --stage 6 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.fromstage6.log # after fixing the error from the right above
## Other backend experiments with different spkloss_weight (RUNNING:0.001 | DONE: 3, 0.01, 0.03, 0.3, 0 | NOT RUN YET:)
bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight[spkloss_weight] 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight[spkloss_weight].log
bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly.log # for spkidonly
(DONE) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_unsync200to400random.log # for spkidonly
(DONE: 6.67, spkloss_best) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.log # for spkidonly
(DONE: 6.67, spkloss_best) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --model snapshot.ep.10 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.spkloss_best.snapshot.ep.10.log # for spkidonly
(DONE: 5.59, spkacc_best) bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.28--nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.spkacc_best.snapshot.ep.28.log # for spkidonly
(DONE: 5.29, spkacc_best after all 50 epochs) bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.48 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.spkacc_best.snapshot.ep.48.log # for spkidonly
(DONE: 5.75, alllossesbest, ep.49) bash run_backendonly_cpuparallel.voxceleb1.fbankonly.sh --model snapshot.ep.49 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_fbankonly_pytorch_fbankonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_fbankonly_pytorch_fbankonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.spkacc_best.snapshot.ep.49.log # for spkidonly
### (+ add specaug)
(ING: spklossbest (ep.48)) bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.48 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_spkidonly_shufflebatching_bs128_ttslossw0_spklossw1_final_unsync_specaugrate0.5/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_spkidonly_shufflebatching_bs128_ttslossw0_spklossw1_final_unsync_specaugrate0.5.spklossbest.snapshot.ep.48.log
(ING: spklossNspkaccbest (ep.50)) bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.50 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_spkidonly_shufflebatching_bs128_ttslossw0_spklossw1_final_unsync_specaugrate1.0/ 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_spkidonly_shufflebatching_bs128_ttslossw0_spklossw1_final_unsync_specaugrate1.0.spklossbest.snapshot.ep.50.log
## (*** Wrong SV eval due to the wrong normalization above*** ) run the backend exps - end

## *** SV eval after voxceleb1_test norm fixed - start
### spkidonly
#### whole utt training
(DONE: 5.34, spklossbest) bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly.lossbest.log
(DONE: 5.04 spkaccbest, ep.48) bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.48 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly.spkaccbest.ep.48.log
#### chunk training
(DONE: 3.90, 5.29 model above) bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --model snapshot.ep.48 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.spkacc_best.snapshot.ep.48.log # for spkidonly
### various lda_dim
#### 5.29 model above
for lda_dim in 150 200 250 300 350 400;do
    (bash run_backendonly_cpuparallel.varlda_dim.voxceleb1.sh --lda_dim ${lda_dim} --stage 6 --model snapshot.ep.48 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.lda_dim${lda_dim}.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_shufflebatching_bs128_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.spkacc_best.snapshot.ep.48.log) &
done
### tts only
#### 13.49 EER model (librispeechXformerASR): log/run.eer_mindcf.voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.log (The log is overwritten)
(RERUN FROM EXTRACTING EMBEDDINGS)
(DONE: 9.78) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0 2>&1 | tee log/backend.voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.possiblylossbest.log
##### various lda_dim
(DONE: 9.70 is the best (w.r.t EER) with lda_dim=250)
for lda_dim in 150 200 250 300 350 400;do
    (bash run_backendonly_cpuparallel.varlda_dim.voxceleb1.sh --lda_dim ${lda_dim} --stage 6 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0 2>&1 | tee log/backend.lda_dim${lda_dim}.voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.possiblylossbest.log) &
done
#### 12.62 EER model (wsjConvBlstmASR): log/run.eer_mindcf.voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03_snapshot.ep.13.log (The log is overwritten)
(DONE: 9.38) bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.lossbest.log
##### various lda_dim
(DONE: 9.27 is the best (w.r.t EER) with lda_dim=250)
for lda_dim in 150 200 250 300 350 400;do
    (bash run_backendonly_cpuparallel.varlda_dim.voxceleb1.sh --lda_dim ${lda_dim} --stage 6 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0 2>&1 | tee log/backend.lda_dim${lda_dim}.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.lossbest.log) &
done
### spkid-tts
#### 6.43 EER model (librispeechXformerASR): log/run.eer_mindcf.voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03_snapshot.ep.13.log (The log is overwritten)
(DONE: 4.32) bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.13 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03 2>&1 | tee log/backend.voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03.spkaccbest.ep.13.log
##### various lda_dim
(DONE: 4,24 is the best (only w.r.t EER) with lda_dim=200)
for lda_dim in 150 200 250 300 350 400;do
    (bash run_backendonly_cpuparallel.varlda_dim.voxceleb1.sh --lda_dim ${lda_dim} --stage 6 --model snapshot.ep.13 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03 2>&1 | tee log/backend.lda_dim${lda_dim}.voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03.spkaccbest.ep.13.log) &
done
#### 6.03 EER model (wsjConvBlstmASR): log/run.eer_mindcf.voxceleb1_train_filtered_train_librispeechXformerASR_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03_snapshot.ep.13.log (The log is overwritten)
(ING) bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.001 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.001.lossbest.log
(ING) bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.01 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.01.lossbest.log
(DONE: 4.53) bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03.lossbest.log
(ING) bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.3 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.3.lossbest.log
(ING) bash run_backendonly_cpuparallel.voxceleb1.withcorrectnorm_voxceleb1_test.sh --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight3 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight3.lossbest.log
##### various lda_dim
(DONE: 4.51 is the best (w.r.t EER) with lda_dim=250)
for lda_dim in 150 200 250 300 350 400;do
    (bash run_backendonly_cpuparallel.varlda_dim.voxceleb1.sh --lda_dim ${lda_dim} --stage 6 --nj 70 --ngpu 0 --expdir exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03 2>&1 | tee log/backend.lda_dim${lda_dim}.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_bs48_spkloss_weight0.03.lossbest.log) &
## *** SV eval after voxceleb1_test norm fixed - end


# (+) Backend evaluation with short utterances (5s, 3s, and 1s) - *** 1) This should be run after training a plda model with embeddings from whole utterances, 2) (FIX FOR THIS. CURRENTLY, THE SAME SEGMENTS WERE USED BUT FEATURES ARE DIFFERENT. TRIALS ARE SAME) Segwise features and trials should be identical to ones used in /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3. Only difference is a embedding extractor ***
## Generate features seg-wise features. Copy the same segment files from ../asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/ for fair comparison
bash gen_short_utterances.sh 2>&1 | tee log/gen_short_utterances.log
## Copy some functions from /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3 (e.g., cp /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/merge_fullNseg.py .)
## run back-end evaluation only
### (DONE) seg500
#### with SPKID ONLY
##### utt-wise SV - shortshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.shortshort_evalonly.sh --eval_set voxceleb1_test_seg500 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final.seg500.shortshort.log
##### utt-wise SV - fullshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.fullshort_evalonly.sh --eval_set voxceleb1_test_seg500 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final.seg500.fullshort.log
##### seg-wise SV - shortshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.shortshort_evalonly.sh --eval_set voxceleb1_test_seg500 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.seg500.shortshort.log
##### seg-wise SV - fullshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.fullshort_evalonly.sh --eval_set voxceleb1_test_seg500 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.seg500.fullshort.log
### (DONE) seg 300
#### with SPKID ONLY
##### utt-wise SV - shortshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.shortshort_evalonly.sh --eval_set voxceleb1_test_seg500_seg300 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final.seg300.shortshort.log
##### utt-wise SV - fullshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.fullshort_evalonly.sh --eval_set voxceleb1_test_seg500_seg300 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final.seg300.fullshort.log
##### seg-wise SV - shortshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.shortshort_evalonly.sh --eval_set voxceleb1_test_seg500_seg300 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.seg300.shortshort.log
##### seg-wise SV - fullshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.fullshort_evalonly.sh --eval_set voxceleb1_test_seg500_seg300 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.seg300.fullshort.log
### (DONE) seg 100
#### with SPKID ONLY
##### utt-wise SV - shortshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.shortshort_evalonly.sh --eval_set voxceleb1_test_seg500_seg300_seg100 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final.seg100.shortshort.log
##### utt-wise SV - fullshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.fullshort_evalonly.sh --eval_set voxceleb1_test_seg500_seg300_seg100 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final.seg100.fullshort.log
##### seg-wise SV - shortshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.shortshort_evalonly.sh --eval_set voxceleb1_test_seg500_seg300_seg100 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.seg100.shortshort.log
##### seg-wise SV - fullshort
(DONE) bash run_backendonly_cpuparallel.voxceleb1.fullshort_evalonly.sh --eval_set voxceleb1_test_seg500_seg300_seg100 --nj 70 --ngpu 0 --expdir /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid/exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random 2>&1 | tee log/backend.voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spkloss_weight1_final_unsync200to400random.seg100.fullshort.log



# Exp.s with only 80-dim features (without 3-dim pitch)
bash run_leaveout_pitch.sh
