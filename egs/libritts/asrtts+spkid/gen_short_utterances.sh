# Copied and edited from ../asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/
# This script extracts feature segments using kaldi bin for the short utterances (5s, 3s, and 1s)

### get 500 segments
. path.sh
oridir=../asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/
eval_set=voxceleb1_test # do not put "slash" in the end
seglen=500

cp ${oridir}/dump/${eval_set}/segments_${seglen} dump/${eval_set}/ # *** Get only segments_${seglen} files from ${oridir} so that the segments used for SV are identical over experiments
mkdir -p dump/${eval_set}_seg${seglen}
extract-feature-segments scp:dump/${eval_set}/feats.scp dump/${eval_set}/segments_${seglen} ark,scp:dump/${eval_set}_seg${seglen}/feats.ark,dump/${eval_set}_seg${seglen}/feats.scp
feat-to-len scp:dump/${eval_set}_seg${seglen}/feats.scp ark,t:dump/${eval_set}_seg${seglen}/utt2num_frames # this is for the main script to be run after this prep. script
### get 300 segments
eval_set=voxceleb1_test_seg500 # do not put "slash" in the end
seglen=300
cp ${oridir}/dump/${eval_set}/segments_${seglen} dump/${eval_set}/ # *** Get only segments_${seglen} files from ${oridir} so that the segments used for SV are identical over experiments
mkdir -p dump/${eval_set}_seg${seglen}
extract-feature-segments scp:dump/${eval_set}/feats.scp dump/${eval_set}/segments_${seglen} ark,scp:dump/${eval_set}_seg${seglen}/feats.ark,dump/${eval_set}_seg${seglen}/feats.scp
feat-to-len scp:dump/${eval_set}_seg${seglen}/feats.scp ark,t:dump/${eval_set}_seg${seglen}/utt2num_frames # this is for the main script to be run after this prep. script

### get 100 segments
eval_set=voxceleb1_test_seg500_seg300 # do not put "slash" in the end
seglen=100
cp ${oridir}/dump/${eval_set}/segments_${seglen} dump/${eval_set}/ # *** Get only segments_${seglen} files from ${oridir} so that the segments used for SV are identical over experiments
mkdir -p dump/${eval_set}_seg${seglen}
extract-feature-segments scp:dump/${eval_set}/feats.scp dump/${eval_set}/segments_${seglen} ark,scp:dump/${eval_set}_seg${seglen}/feats.ark,dump/${eval_set}_seg${seglen}/feats.scp
feat-to-len scp:dump/${eval_set}_seg${seglen}/feats.scp ark,t:dump/${eval_set}_seg${seglen}/utt2num_frames # this is for the main script to be run after this prep. script
