. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

eval_set=voxceleb1_test_seg[seglen]
embname= # e.g) train_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spklossw1_final

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

echo evaluation set: ${eval_set}
echo embname: ${embname}
trial_path=dump/${eval_set}/trials_fullshort # for now it is fixed all the time
score_path=exp/scores_voxceleb1_${eval_set}_embname${embname}.fullshort # path for the calculated scores given trials
python /export/b17/janto/SRE18/v1.8k/scoring_software/scoring.py ${trial_path} ${score_path} 2>&1 | tee log/run.eer_mindcf.${embname}.${eval_set}.fullshort.log
