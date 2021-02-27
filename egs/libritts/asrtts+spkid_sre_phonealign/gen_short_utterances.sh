# This script generates kaldi segments files for the short utterances (5s, 3s, and 1s)

## Assuming feat-to-len was run already okay (althought it was not, it would not cause a big problems since the output difference from feat-to-len is small)
### get 500 segments and create utt2num_frames from the segments file
. path.sh
eval_set=voxceleb1_test # do not put "slash" in the end
seglen=500
python gen_shortutt_seg.py dump/${eval_set}/utt2num_frames ${seglen} 2>&1 | tee dump/${eval_set}/gen_shortutt_seg.${seglen}.log
mkdir -p dump/${eval_set}_seg${seglen}
extract-feature-segments scp:dump/${eval_set}/feats.scp dump/${eval_set}/segments_${seglen} ark,scp:dump/${eval_set}_seg${seglen}/feats.ark,dump/${eval_set}_seg${seglen}/feats.scp
feat-to-len scp:dump/${eval_set}_seg${seglen}/feats.scp ark,t:dump/${eval_set}_seg${seglen}/utt2num_frames
### get 300 segments
eval_set=voxceleb1_test_seg500 # do not put "slash" in the end
seglen=300
python gen_shortutt_seg.py dump/${eval_set}/utt2num_frames ${seglen} 2>&1 | tee dump/${eval_set}/gen_shortutt_seg.${seglen}.log
mkdir -p dump/${eval_set}_seg${seglen}
extract-feature-segments scp:dump/${eval_set}/feats.scp dump/${eval_set}/segments_${seglen} ark,scp:dump/${eval_set}_seg${seglen}/feats.ark,dump/${eval_set}_seg${seglen}/feats.scp
feat-to-len scp:dump/${eval_set}_seg${seglen}/feats.scp ark,t:dump/${eval_set}_seg${seglen}/utt2num_frames

### get 100 segments
eval_set=voxceleb1_test_seg500_seg300 # do not put "slash" in the end
seglen=100
python gen_shortutt_seg.py dump/${eval_set}/utt2num_frames ${seglen} 2>&1 | tee dump/${eval_set}/gen_shortutt_seg.${seglen}.log
mkdir -p dump/${eval_set}_seg${seglen}
extract-feature-segments scp:dump/${eval_set}/feats.scp dump/${eval_set}/segments_${seglen} ark,scp:dump/${eval_set}_seg${seglen}/feats.ark,dump/${eval_set}_seg${seglen}/feats.scp
feat-to-len scp:dump/${eval_set}_seg${seglen}/feats.scp ark,t:dump/${eval_set}_seg${seglen}/utt2num_frames
