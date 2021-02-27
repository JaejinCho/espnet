# Copied from ../asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/run.eer_mindcf.varlda_dim.voxceleb1.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

lda_dim=150
eval_set=voxceleb1_test
embname= # e.g) train_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spklossw1_final

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

echo lda dimension: ${lda_dim}
echo evaluation set: ${eval_set}
echo embname: ${embname}
trial_path=data/${eval_set}/trials # for now it is fixed all the time
score_path=exp/scores_voxceleb1_${eval_set}_lda_dim${lda_dim}.embname${embname} # path for the calculated scores given trials
python /export/b17/janto/SRE18/v1.8k/scoring_software/scoring.py ${trial_path} ${score_path} 2>&1 | tee log/run.eer_mindcf.lda_dim${lda_dim}.embname_${embname}.evalset_${eval_set}.log
