# Edited from run.dataaug.check.sh
# To final check for one corpus if this script runs as expected

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=-1
stop_stage=100

# feature extraction related
fs=8000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_fft=256    # number of fft points (~32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=80   # number of shift points (~10ms)
win_length="" # window length
n_mels=64     # number of mel basis
train_set=sre16_eval_tr60_yue # corpus name in sre20 recipe


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

musan_root=/export/corpora5/JHU/musan
kaldi_datadir=/export/b16/jcho/hyperion_trials/egs/sre20-cts/v1/data
mkdir -p data/${train_set}_hyperion_prep # hyperion_prep means "prepared data dir by hyperion"
cp ${kaldi_datadir}/${train_set}/{utt2spk,spk2utt,utt2num_frames} data/${train_set}_hyperion_prep && cp data/${train_set}/wav.scp data/${train_set}_hyperion_prep # cp wav.scp from data/${train_set}/ since some wav list is removed after SAD from ${kaldi_datadir}/${train_set}/. This causes some error later in the utils/validate_data_dir.sh step in utils/copy_data_dir.sh
utils/fix_data_dir.sh data/${train_set}_hyperion_prep
# *** NOTE ***: data/${train_set} & data/${train_set}_* & data/${train_set}_hyperion_prep are all different directories

# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Data augmentation"
  calc(){ awk "BEGIN { print "$*" }"; }
  frame_shift=`calc ${n_shift}/${fs}`
  echo "Calculated frame shift: ${frame_shift} sec"

  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/${train_set}_hyperion_prep/utt2num_frames > data/${train_set}_hyperion_prep/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate ${fs} \
    data/${train_set}_hyperion_prep data/${train_set}_reverb

  cp data/${train_set}/{segments,spk2utt,utt2spk} data/${train_set}_reverb
  rm data/${train_set}_reverb/utt2uniq # (TODO: check if this is fine. For now, it is to avoid an error)
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/${train_set}_reverb data/${train_set}_reverb.new
  rm -rf data/${train_set}_reverb
  mv data/${train_set}_reverb.new data/${train_set}_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  if [ ! -f steps/data/make_musan.sh ]; then
    kaldi_voxceleb_dir=/export/c07/jcho/kaldi/egs/voxceleb/v2
    cp ${kaldi_voxceleb_dir}/steps/data/make_musan.sh steps/data
    cp ${kaldi_voxceleb_dir}/steps/data/make_musan.py steps/data

  fi

  # ********* Remove data/musan first before running the below block if the data sampling-rate changes (e.g. I changed them with "_pre" suffices)
  if [ ! -d data/musan ]; then
    #local/make_musan.sh $musan_root data # commit-id 461b50c2c8d219c31eaa67fdb00587be0374a170 (2019 Mar)
    steps/data/make_musan.sh --sampling-rate ${fs} $musan_root data # more recent version with commit-id 92191b6e788aa7016625fadda55da083b8eed027 (2021 Feb 09)
  fi

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    if [ ! -s data/musan_${name}/reco2dur ]; then
        utils/data/get_utt2dur.sh data/musan_${name}
        mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    fi
  done

 # Before this stage, I ran "grep -Rl "/export/corpora/" data/musan* | xargs sed -i 's!/export/corpora/!/export/corpora5/!g' # reference: https://naysan.ca/2020/06/02/linux-command-line-find-replace-in-multiple-files/" since /export/corpora/ now is /export/corpora5/
 # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "suffix2bremoved" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/${train_set}_hyperion_prep data/${train_set}_noise
  grep -Rl "suffix2bremoved" data/${train_set}_noise | xargs sed -i 's/-suffix2bremoved//g' # By this line, only difference in files bewteen data/${train_set} and data/${train_set}_noise is wav.scp contents
  cp data/${train_set}/{segments,spk2utt,utt2spk} data/${train_set}_noise
  utils/copy_data_dir.sh --utt-suffix "-noise" data/${train_set}_noise data/${train_set}_noise.new
  rm -rf data/${train_set}_noise
  mv data/${train_set}_noise.new data/${train_set}_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "suffix2bremoved" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${train_set}_hyperion_prep data/${train_set}_music
  grep -Rl "suffix2bremoved" data/${train_set}_music | xargs sed -i 's/-suffix2bremoved//g' # By this line, only difference in files bewteen data/${train_set} and data/${train_set}_music is wav.scp contents
  cp data/${train_set}/{segments,spk2utt,utt2spk} data/${train_set}_music
  utils/copy_data_dir.sh --utt-suffix "-music" data/${train_set}_music data/${train_set}_music.new
  rm -rf data/${train_set}_music
  mv data/${train_set}_music.new data/${train_set}_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "suffix2bremoved" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/${train_set}_hyperion_prep data/${train_set}_babble
  grep -Rl "suffix2bremoved" data/${train_set}_babble | xargs sed -i 's/-suffix2bremoved//g' # By this line, only difference in files bewteen data/${train_set} and data/${train_set}_babble is wav.scp contents
  cp data/${train_set}/{segments,spk2utt,utt2spk} data/${train_set}_babble
  utils/copy_data_dir.sh --utt-suffix "-babble" data/${train_set}_babble data/${train_set}_babble.new
  rm -rf data/${train_set}_babble
  mv data/${train_set}_babble.new data/${train_set}_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/${train_set}_aug data/${train_set}_reverb data/${train_set}_noise data/${train_set}_music data/${train_set}_babble
fi
