# Edited from ../asrtts+spkid_voxceleb1_kaldiasr_phonealign_rf3/run.eer_mindcf.voxceleb1.flextrial.sh

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

trial_list=''
embname='' # e.g) train_train_pytorch_train_pytorch_tacotron2+spkemb_spkidonly_speakeridonly_ttslossw0_spklossw1_final

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

echo evaluation set: ${trial_list}
echo embname: ${embname}
trial_set=`basename ${trial_list}`
score_path=exp/scores_voxceleb2train_trialset${trial_set}_embname${embname}
python /export/b17/janto/SRE18/v1.8k/scoring_software/scoring.py ${trial_list} ${score_path} 2>&1 | tee log/run.eer_mindcf.voxceleb2backendtrain.trialset${trial_set}_embname${embname}.log
