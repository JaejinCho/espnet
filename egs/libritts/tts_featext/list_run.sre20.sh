- (ING) sre20 exp
    < RERUN WITH segments FILES USING SAD >

    - spkid lab 9 corpora (Ran for sre16_eval_tr60_yue first as a test (run.FE_8k.sre16_eval_tr60_yue.sadseg.sh) and then ran for all other corpora. Next time, data prep. results could be slightly different since I reflect things here and there. However, the current prepared data look okay so keep with them for further processes for now)
    # (DONE: stage 0 only. CHECKED THE RESULTS AND THEN RAN STAGE 1. datadir should include "wav.scp,utt2spk,spk2utt" and datadir_seg "segments" ) bash run.FE_8k.sre_swbd_fisherspa_voxcelebtel.sadseg.sh --nj 70 --stage 0 --stop_stage 0 2>&1 | tee log/run.FE_8k.sre_swbd_fisherspa_voxcelebtel.sadseg.onlystage0.log
        - Fixes (These are all reflected in the run):
            1) (DONE. TODO (not for now) left) Some machines did not have some libaries. (So limited to some c machines when I qsub. This is a stopgap (TODO: The actual solution is to compile Kaldi again properly to updated/changed CLSP grid env.))
            2) (DONE) Extra processes are needed after getting SAD segments: (1) cat or delete some segments, (2) With the new segments, create new utt2spk and spk2utt.
                - (DONE) The corresoponding .py scripts were
                  written in /export/b11/jcho/kaldi_20190316/egs/librispeech/s5/SAD_related/vimal_SAD/outdatadir_seg/{seg2newseg.py,utt2spk2newutt2spk.py} >> copied to local/
            3) (DONE) For voxcelebcat_tel, I need to set read_entire_file=true for get_utt2dur*.sh so I run again only for the corpus after removing the data dir generated imperfectly.
                - (DONE) bash run.FE_8k.sre_swbd_fisherspa_voxcelebtel.sadseg.sh --nj 70 --stage 0 --stop_stage 0 2>&1 | tee log/run.FE_8k.sre_swbd_fisherspa_voxcelebtel.sadseg.onlystage0.voxcelebcat_tel_only_again.log
            4) (DONE) For sre18_cmn2_train_lab and voxcelebcat_tel, creating a new segments did not run properly due to some typo in the code, local/seg2newseg.py. Now it is fixed.
            5) (DONE) For sre18_cmn2_train_lab, data/sre18_cmn2_train_lab_whole_hires/ dir does not include things included in other copora but I IGONORE THIS FOR NOW since the final segements look correct
            6) (DONE) fix_data_dir.sh added in the end of stage 0
    # (DONE, stage 1 only) bash run.FE_8k.sre_swbd_fisherspa_voxcelebtel.sadseg.sh --nj 70 --stage 1 --stop_stage 1 2>&1 | tee log/run.FE_8k.sre_swbd_fisherspa_voxcelebtel.sadseg.onlystage1.log
    # (ING, all stages for only sre16_eval_tr60_yue since its ark files are gone for some reason) bash run.FE_8k.sre_swbd_fisherspa_voxcelebtel.sadseg.sh --nj 70 2>&1 | tee log/run.FE_8k.sre_swbd_fisherspa_voxcelebtel.sadseg.only4sre16_eval_tr60_yue.log


    - NO spkid lab 25 babel corpora (********* NOTE ***********: I can run below (for babel) together with the above but for now, just run separately like below)
    # (DONE: stage 0 only. CHECKED THE RESULTS AND THEN RAN STAGE 1. datadir should include "wav.scp,utt2spk,spk2utt" and datadir_seg "segments" ) bash run.FE_8k.babel_25langs.sadseg.sh --nj 70 --stage 0 --stop_stage 0 2>&1 | tee log/run.FE_8k.babel_25langs.sadseg.onlystage0.log
        - Fixes:
            1) (DONE) Some errors happened for 8 lang.s (due to # uttids mismatch in segments and wav.scp) --> fixed it by fix_data_dir.sh. now the part is included in the script
                data/babel_bengali
                data/babel_tokpisin
                data/babel_haitian
                data/babel_telugu
                data/babel_zulu
                data/babel_tagalog
                data/babel_pashto
                data/babel_tamil
    # (DONE) bash run.FE_8k.babel_25langs.sadseg.sh --nj 70 --stage 0 --stop_stage 1 2>&1 | tee log/run.FE_8k.babel_25langs.sadseg.untilstage1.log
       (DONE again for above 8 lang.s: stage 1 only) bash run.FE_8k.babel_25langs.sadseg.sh --nj 70 --stage 1 --stop_stage 1 2>&1 | tee log/run.FE_8k.babel_25langs.sadseg.onlystage1.log


    - eval 8 corpora (sre16_eval40_yue_enroll sre16_eval40_yue_test sre16_eval40_tgl_enroll sre16_eval40_tgl_test sre19_eval_enroll_cmn2 sre19_eval_test_cmn2 sre20cts_eval_enroll sre20cts_eval_test)
    # (DONE) bash run.FE_8k.evalcorpora.sadseg.sh --nj 70 --stage 0 --stop_stage 1 2>&1 | tee log/run.FE_8k.evalcorpora.sadseg.untilstage1.log

    - *** After the runs above for "spkid lab 9 corpora", "NO spkid lab 25 babel corpora", and "eval 8 corpora" (after the final feature extraction),
    I ran "fix_data_dir.sh" on every corpus for sanity check. (The line is now added to all the scripts above)
    ---> NO CHANGE after this run for ALL THE CORPORA

    - Sanity check to see if # lines in wav.scp and spk2utt files stay the same after the whole process above (BEFORE the run below simply set data_list according to the subset):
    # for dname in `echo $data_list`;do echo $dname; num_wav_ori=`wc -l $datadir_ori/$dname/wav.scp | awk '{print $1}'`; num_wav_processed=`wc -l data/$dname/wav.scp | awk '{print $1}'`; num_spk_ori=`wc -l $datadir_ori/$dname/spk2utt | awk '{print $1}'`; num_spk_processed=`wc -l data/$dname/spk2utt | awk '{print $1}'`; if [ $num_wav_ori -ne $num_wav_processed ]; then echo "Different after process in num_wav"; fi; if [ $num_spk_ori -ne $num_spk_processed ]; then echo "Different after process in num_spk"; fi; done
    # (SOME CORPORA that changed # lines in wav.scp or spk2utt:
        - Training: 1) Among lab data, "sre_tel sre16_train_dev_ceb sre16_train_dev_cmn sre18_cmn2_train_lab swbd voxcelebcat_tel", 2) Among nonlab (babel) data, "babel_bengali babel_haitian babel_pashto babel_tagalog babel_tamil babel_telugu babel_tokpisin babel_zulu"
            > There would be ******* NO ******* problems since above corpora are used for training front-end or back-end.
        - Eval (DUE TO THIS CORPORA, THERE POSSIBLY IS SOME ERROR IN EVAL) sre19_eval_test_cmn2

    - (*** (FOR BACKEND FOR NOW) NOT TO DISCARD ANY SEGMENTS FOR LATER PROCESSES just for 2 corpora: sre19_eval_enroll_cmn2, sre19_eval_test_cmn2)
     # (THIS IS NOT ENOUGH. RUN BELOW (Adding NOdiscard in some dir names)) bash run.FE_8k.sre19_eval_cmn2.sadseg.sh --nj 70 --stage 0 --stop_stage 1 2>&1 | tee log/run.FE_8k.sre19_eval_cmn2.sadseg.untilstage1.log
     bash run.FE_8k.sre19_eval_cmn2.sadseg.keepallutts.sh --nj 70 --stage 0 --stop_stage 1 2>&1 | tee log/run.FE_8k.sre19_eval_cmn2.sadseg.keepallutts.untilstage1.log
     # extend to all evalcorpora
     bash run.FE_8k.evalcorpora.sadseg.keepallutts.sh --nj 70 2>&1 | tee log/run.FE_8k.evalcorpora.sadseg.keepallutts.log

    - (ING) (KALDI-STYLE (voxceleb/v2 recipe) DATA AUGMENTATION)
        1. (ING) EXACT KALDI-STYLE
        (DONE) 1st, check with just 3 corpora (sre16_eval_tr60_yue babel_pashto sre16_eval40_yue_enroll) -> It works (I manually listened to the augmented wav files)
        $ bash run.dataaug.finalcheck.sh --train_set ${train_set} 2>&1 | tee log/run.dataaug.finalcheck.${train_set}.log # train_set was either sre16_eval_tr60_yue or babel_pashto or sre16_eval40_yue_enroll
        (DONE. Check needed if it ran as expected) 2nd, run for all except eval corpora
        $ bash run.kaldidataaug.sre20.sh 2>&1 | tee log/run.kaldidataaug.sre20.log
        2. VARIANT (as in SELF-SUPERVISED TEXT-INDEPENDENT SPEAKER VERIFICATION USING PROTOTYPICAL MOMENTUM CONTRASTIVE LEARNING)






    < BELOW scripts WERE DONE RUNNING AND REMOVED >
    (DONE) bash run.FE_8k.sre_swbd_voxceleb.sh 2>&1 | tee log/run.FE_8k.sre_swbd_voxceleb.log
    (DONE) bash run.FE_8k.fisher_spa.sh 2>&1 | tee log/run.FE_8k.fisher_spa.log
    (DONE) bash run.FE_8k.voxcelebcat_tel.sh 2>&1 | tee log/run.FE_8k.voxcelebcat_tel.log
