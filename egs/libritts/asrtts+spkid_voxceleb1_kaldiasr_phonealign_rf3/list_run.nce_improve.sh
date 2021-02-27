# Outline: front-end and back-end experiments for nce_improve (moco) included in training
## frontend
(Old example copied from list_run.2.sh) bash run.spkidonly_nceloss.sh --train_config conf/train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76.yaml --nceloss_weight 1 --ngpu 1 --ttsloss_weight 0 --spkloss_weight 0 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.onlynceloss_improve_reportnceval.shufflebatching.bs76.chunks.from200to400.stage4only.log
### (fix) from above to include nceloss for validation metrics at the end of every epoch
(ING) bash run.spkidonly_nceloss.sh --train_config conf/train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs80_ep150.yaml --nceloss_weight 1 --ngpu 1 --ttsloss_weight 0 --spkloss_weight 0 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.onlynceloss_improve_reportnceval.shufflebatching.bs80.ep150.chunks.from200to400.stage4only.log
### (fix) from above to include shuffle BN for one gpu
(ERROR: CUDA OOM) bash run.spkidonly_nceloss.sh --train_config conf/train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shuffleBN_reportnceval_shufflebatching_bs128.yaml --nceloss_weight 1 --ngpu 1 --ttsloss_weight 0 --spkloss_weight 0 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.onlynceloss_improve_shuffleBN_reportnceval.shufflebatching.bs128.chunks.from200to400.stage4only.log
(DONE, changed bs 128 to 80 due to CUDA OOM) bash run.spkidonly_nceloss.sh --train_config conf/train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shuffleBN_reportnceval_shufflebatching_bs80_ep150.yaml --nceloss_weight 1 --ngpu 1 --ttsloss_weight 0 --spkloss_weight 0 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.onlynceloss_improve_shuffleBN_reportnceval.shufflebatching.bs80.ep150.chunks.from200to400.stage4only.log # first run was WRONG since I did not shuffle inputs for key encoder, thus changing the resulting exp/* into exp/*_WRONG_noshuffle4keyencoder & (DONE) THEN RERUNNDONE after fixing it (with ep150 straight)
### (fix) from above to add a function to reduce the overlap between random chunks from a same utterance (Refernce: The IDLAB VoxCeleb Speaker Recognition Challenge 2020 SystemDescription) & also to save model.loss.best with nceloss best one in validation
(ING) bash run.spkidonly_nceloss.sh --train_config conf/train_tacotron2+spkemb_spkidonly_nceloss_improve_shuffleBN_reportnceval_shufflebatch_bs80_ep150_reduceoverlap.yaml --nceloss_weight 1 --ttsloss_weight 0 --spkloss_weight 0 --stage 4 --n_average 0 --stop_stage 4 2>&1 | tee log/run.onlynceloss_improve_shuffleBN_reportnceval_reduceoverlap.shufflebatching.bs80.ep150.chunks.from200to400.stage4only.log










## Backend
expdirname=exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight1_final_nceloss_weight0.01_unsync200to400random
expdirname_short=`echo $expdirname | sed -e 's/pytorch_train_pytorch_tacotron2+spkemb_//g'`
mv ${expdirname} ${expdirname_short} # the original expdirname is too long
(Example) bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir ${expdirname_short} 2>&1 | tee log/backend.$(basename ${expdirname_short}).log # in the log by "cut", I delete some part since it is too long to generate
### The above command ran for all exp dirs below:
(DONE) exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight1_final_nceloss_weight0.01_unsync200to400random
(DONE) exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random
(DONE) exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight1_final_nceloss_weight0.01_unsync200to400random
(DONE) exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random
(DONE) exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight1_final_nceloss_weight1_unsync200to400random
(DONE) exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight1_final_nceloss_weight0.1_unsync200to400random
(DONE) exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight1_final_nceloss_weight1_unsync200to400random
(DONE) exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight1_final_nceloss_weight0.1_unsync200to400random
#### after finishing 50 epochs (for only nceloss experiments)
expdirname_short=[one of exps below]
bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir ${expdirname_short} 2>&1 | tee log/backend.$(basename ${expdirname_short}).log
##### exps list - start
exp/voxceleb1_train_filtered_train_pytorch_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_K12160_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random
exp/voxceleb1_train_filtered_train_pytorch_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_K1216_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random
exp/voxceleb1_train_filtered_train_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random
exp/voxceleb1_train_filtered_train_spkidonly_nceloss_improve_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random
##### exps list - end

### additionally, below command were run (model trained with vox2 data)
expdirname_short=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_spkloss_weight0_nceloss_weight1_unsync
bash run_backendonly_cpuparallel.voxceleb1.sh --nj 70 --ngpu 0 --expdir ${expdirname_short} 2>&1 | tee log/backend.$(basename ${expdirname_short}).log
bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.6 --nj 70 --ngpu 0 --expdir ${expdirname_short} 2>&1 | tee log/backend.$(basename ${expdirname_short}).log
bash run_backendonly_cpuparallel.voxceleb1.normdecoding.sh --model snapshot.ep.6 --nj 70 --ngpu 0 --expdir ${expdirname_short} 2>&1 | tee log/backend.$(basename ${expdirname_short})_embnorm.log # with emb normalization in decoding
bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.8 --nj 70 --ngpu 0 --expdir ${expdirname_short} 2>&1 | tee log/backend.$(basename ${expdirname_short}).log

### for shuffleBN ep150 experiment
#### make the expdir name shorter
expdirname=exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shuffleBN_reportnceval_shufflebatching_bs80_ep150_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random/
expdirname_short=`echo $expdirname | sed -e 's/pytorch_train_pytorch_tacotron2+spkemb_//g'`
mv ${expdirname} ${expdirname_short}
expdirname_short_short=exp/voxceleb1_train_filtered_train_spkidonly_nceloss_improve_shuffleBN_reportnceval_bs80_ep150_ttslossw0_spkloss_w0_nceloss_w1_unsync
mv ${expdirname_short} ${expdirname_short_short}
##### exps list - start
###### vox2 - start (******** NOTE ********: This moved back to the original expdirname to resume the training again)
expdirname=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shuffleBN_reportnceval_shufflebatching_bs80_spkloss_weight0_nceloss_weight1_unsync
expdirname_short_short=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_nce_improve_shuffleBN_reportnceval_shufflebatch_bs80_spkloss_weight0_nceloss_weight1_unsync
mv ${expdirname} ${expdirname_short_short}
mv ${expdirname_short_short} ${expdirname}
####### (above) results with ep.13 (the best nceloss model at the time): EER 16.09 (NO big difference with one trained with only vox1 so I am running this model for more epochs)
expdirname=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shuffleBN_reportnceval_shufflebatching_bs80_spkloss_weight0_nceloss_weight1_unsync
expdirname_short_short=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_nceloss_improve_shuffleBN_reportnceval_shufflebatch_bs80_spkloss_weight0_nceloss_weight1_unsync
mv ${expdirname} ${expdirname_short_short}
bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.50 --nj 70 --ngpu 0 --expdir ${expdirname_short_short} 2>&1 | tee log/backend.$(basename ${expdirname_short_short}).ep50.ncelossbest.log
EER: 20.20 DCF5e-2: 0.797 / 0.929 DCF1e-2: 0.833 / 1.544 DCF5e-3: 0.838 / 2.056 DCF1e-3: 0.838 / 4.551

expdirname=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_shuffleBN_reportnceval_shufflebatching_bs80_reduceoverlap_spkloss_weight0_nceloss_weight1_unsync
expdirname_short_short=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_nceloss_improve_shuffleBN_reportnceval_shufflebatch_bs80_reduceoverlap_spkloss_weight0_nceloss_weight1_unsync
mv ${expdirname} ${expdirname_short_short}
bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.32 --nj 70 --ngpu 0 --expdir ${expdirname_short_short} 2>&1 | tee log/backend.$(basename ${expdirname_short_short}).ep32.ncelossbest.log

expdirname=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs80_spkloss_weight0_nceloss_weight1_unsync
expdirname_short_short=/export/b14/jcho/espnet3/egs/libritts/asrtts+spkid_voxceleb2_phonealign/exp/voxceleb2_train_nceloss_improve_reportnceval_shufflebatch_bs80_spkloss_weight0_nceloss_weight1_unsync
mv ${expdirname} ${expdirname_short_short}
bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.7 --nj 70 --ngpu 0 --expdir ${expdirname_short_short} 2>&1 | tee log/backend.$(basename ${expdirname_short_short}).ep7.ncelossbest.log
###### vox2 - end
##### exps list - end

### (150 ep) run plda backend for all models by every 20 epoch from the best nceloss model 
expdirname=[one of the items in exps list below]
rank_idx=20
for ep_idx in `shownce ${expdirname} | awk 'NR % 20 == 0' | awk '{print $1}'`;do
    bash run_backendonly_cpuparallel.voxceleb1.sh --model snapshot.ep.${ep_idx} --nj 70 --ngpu 0 --expdir ${expdirname} 2>&1 | tee log/backend.$(basename ${expdirname}).ep${ep_idx}.${rank_idx}th_nceloss.log
    rank_idx=$((rank_idx+20))
done
#### exps list - start
(DONE) exp/voxceleb1_train_filtered_train_spkidonly_nceloss_improve_shuffleBN_reportnceval_bs80_ep150_ttslossw0_spkloss_w0_nceloss_w1_unsync # MoCo with shuffleBN (the best model was ep.138 which was already run)
--- results after the above exp (block start) ---
1st best model
EER: 16.61 DCF5e-2: 0.782 / 0.940 DCF1e-2: 0.823 / 1.689 DCF5e-3: 0.828 / 2.362 DCF1e-3: 0.871 / 5.368
20th best model
snapshot.ep.99
EER: 16.09 DCF5e-2: 0.773 / 0.921 DCF1e-2: 0.818 / 1.680 DCF5e-3: 0.831 / 2.259 DCF1e-3: 0.873 / 4.835
40th best model
snapshot.ep.87
EER: 15.79 DCF5e-2: 0.767 / 0.916 DCF1e-2: 0.826 / 1.590 DCF5e-3: 0.837 / 2.165 DCF1e-3: 0.879 / 4.571
60th best model
snapshot.ep.71
EER: 15.57 DCF5e-2: 0.763 / 0.913 DCF1e-2: 0.821 / 1.643 DCF5e-3: 0.835 / 2.129 DCF1e-3: 0.878 / 4.199
80th best model
snapshot.ep.40
EER: 15.89 DCF5e-2: 0.769 / 0.913 DCF1e-2: 0.809 / 1.548 DCF5e-3: 0.828 / 2.119 DCF1e-3: 0.898 / 3.995
100th best model
snapshot.ep.27
EER: 16.96 DCF5e-2: 0.772 / 0.942 DCF1e-2: 0.815 / 1.551 DCF5e-3: 0.833 / 1.853 DCF1e-3: 0.876 / 3.687
120th best model
snapshot.ep.31
EER: 16.65 DCF5e-2: 0.779 / 0.961 DCF1e-2: 0.809 / 1.610 DCF5e-3: 0.819 / 2.068 DCF1e-3: 0.874 / 4.007
140th best model
(ERROR due to which I think device error in grid) snapshot.ep.149
--- results after the above exp (block end) ---

(DONE) exp/voxceleb1_train_filtered_train_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs76_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random # MoCo W/O shuffleBN (the best model is ep.29, which ran separately). RESULTS (EER): 1st: 18.52, 20th: 18.69, 40th: 21.56
##### long expdirname change - start
expdirname_long=exp/voxceleb1_train_filtered_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_nceloss_improve_reportnceval_shufflebatching_bs80_ep150_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random
expdirname=exp/voxceleb1_train_filtered_train_nceloss_improve_reportnceval_shufflebatch_bs80_ep150_ttslossw0_spklossw0_final_ncelossw1_unsync200to400random
mv ${expdirname_long} ${expdirname}
##### long expdirname change - end
(DONE) ${expdirname}
--- results after the above exp (block start) ---
1st best model
snapshot.ep.54
EER: 18.84 DCF5e-2: 0.804 / 1.039 DCF1e-2: 0.834 / 1.756 DCF5e-3: 0.836 / 2.285 DCF1e-3: 0.836 / 4.666
20th best model
snapshot.ep.50
EER: 18.80 DCF5e-2: 0.805 / 1.032 DCF1e-2: 0.834 / 1.762 DCF5e-3: 0.835 / 2.319 DCF1e-3: 0.835 / 4.718
40th best model
snapshot.ep.58
EER: 19.11 DCF5e-2: 0.807 / 1.042 DCF1e-2: 0.834 / 1.791 DCF5e-3: 0.837 / 2.390 DCF1e-3: 0.837 / 4.716
60th best model
snapshot.ep.97
EER: 18.87 DCF5e-2: 0.803 / 1.054 DCF1e-2: 0.840 / 1.997 DCF5e-3: 0.840 / 2.633 DCF1e-3: 0.840 / 6.356
80th best model
snapshot.ep.67
EER: 19.09 DCF5e-2: 0.807 / 1.061 DCF1e-2: 0.830 / 1.877 DCF5e-3: 0.835 / 2.498 DCF1e-3: 0.841 / 5.195
100th best model
snapshot.ep.13
EER: 21.83 DCF5e-2: 0.823 / 1.106 DCF1e-2: 0.858 / 1.822 DCF5e-3: 0.871 / 2.300 DCF1e-3: 0.901 / 3.537
120th best model
snapshot.ep.146
EER: 19.79 DCF5e-2: 0.809 / 1.113 DCF1e-2: 0.838 / 2.115 DCF5e-3: 0.848 / 2.803 DCF1e-3: 0.850 / 7.363
140th best model
snapshot.ep.8
EER: 25.02 DCF5e-2: 0.845 / 1.133 DCF1e-2: 0.890 / 1.603 DCF5e-3: 0.907 / 1.820 DCF1e-3: 0.917 / 2.459
--- results after the above exp (block end) ---

##### long expdirname change - start
expdirname_long=exp/voxceleb1_train_filtered_train_pytorch_train_tacotron2+spkemb_spkidonly_nceloss_improve_shuffleBN_reportnceval_shufflebatch_bs80_ep150_reduceoverlap_speakeridonly_ttslossw0_spkloss_weight0_final_nceloss_weight1_unsync200to400random/
expdirname=exp/voxceleb1_train_filtered_train_nceloss_improve_shuffleBN_reportnceval_shufflebatch_bs80_ep150_reduceoverlap_ttslossw0_spkloss_w0_nceloss_w1_unsync200to400/
mv ${expdirname_long} ${expdirname}
##### long expdirname change - end
(DONE) ${expdirname}
1st best model
snapshot.ep.74
EER: 15.61 DCF5e-2: 0.768 / 0.919 DCF1e-2: 0.827 / 1.641 DCF5e-3: 0.838 / 2.101 DCF1e-3: 0.838 / 3.787
20th best model
EER: 17.00 DCF5e-2: 0.793 / 1.001 DCF1e-2: 0.831 / 1.808 DCF5e-3: 0.844 / 2.440 DCF1e-3: 0.859 / 5.755
40th best model
EER: 16.26 DCF5e-2: 0.779 / 0.961 DCF1e-2: 0.833 / 1.639 DCF5e-3: 0.848 / 2.204 DCF1e-3: 0.849 / 4.532
60th best model
EER: 15.80 DCF5e-2: 0.770 / 0.930 DCF1e-2: 0.831 / 1.489 DCF5e-3: 0.840 / 1.906 DCF1e-3: 0.843 / 3.579
80th best model
EER: 16.54 DCF5e-2: 0.783 / 0.971 DCF1e-2: 0.833 / 1.743 DCF5e-3: 0.838 / 2.353 DCF1e-3: 0.849 / 4.747
100th best model
EER: 15.49 DCF5e-2: 0.773 / 0.932 DCF1e-2: 0.821 / 1.564 DCF5e-3: 0.834 / 2.072 DCF1e-3: 0.834 / 3.791
120th best model
EER: 16.70 DCF5e-2: 0.774 / 0.949 DCF1e-2: 0.826 / 1.518 DCF5e-3: 0.837 / 1.857 DCF1e-3: 0.875 / 2.959
140th best model
EER: 18.97 DCF5e-2: 0.817 / 1.057 DCF1e-2: 0.849 / 1.765 DCF5e-3: 0.858 / 2.120 DCF1e-3: 0.870 / 3.996
#### exps list - end

