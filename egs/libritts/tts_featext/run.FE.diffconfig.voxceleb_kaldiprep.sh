#!/bin/bash

# JJ: copied and edited from run.FE.diffconfig.voxceleb1.sh to prepare voxceleb
# corpus as kaldi preparation in voxceleb/v2/run.sh.
# Things to take care: 1) (DONE) Filter out long sentences,
#                      2) (DONE) Follow kaldi prep but using ESPnet (librosa) FE
#                      3) Divide the result into train and val for NN training
# naming change: 1) (DONE) train/ directory (in kaldi) to voxceleb_train/


# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpu in training
nj=64        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points (~32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=160   # number of shift points (~10ms)
win_length="" # window length

# config files
train_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/60

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set_ori=/export/b13/jcho/espnet_v3/tools/kaldi/egs/voxceleb/v2/data/train/
train_set=voxceleb_train
eval_set=voxceleb1_test
musan_root=/export/corpora/JHU/musan

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    if [ ! -d ${train_set_ori} ]; then
        echo "Generate ${train_set_ori} directory first" && exit 1
    fi
    # train_set (Filter out long utterances for smooth batching): dev_set will
    # be generated after data aug
    if [ ! -s data/${train_set} ]; then
        cp -r ${train_set_ori} data/${train_set}
    fi
    # eval_set
    if [ ! -s data/${eval_set} ]; then
        cp -r /export/b13/jcho/espnet_v3/tools/kaldi/egs/voxceleb/v2/data/voxceleb1_test data/voxceleb1_test
    fi
fi

# JJ add - start
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation (used espnet version NOT kaldi to be consistent with wavform synthesis later using griffi_lim"
    fbankdir=fbank
    for x in ${train_set} ${eval_set};do
        if [ ! -f ${x}/feats.scp ];then
            make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                data/${x} \
                exp/make_fbank/${x} \
                ${fbankdir}
        else
            echo "${x}/feats.scp already exists. Skip generating ${x}/feats.scp"
        fi
    done
    # Do NOT generate VAD decision like Kaldi
    # (empty)
    # Filter out long utterances (legnth calculation is based on feats.scp. If it
    # does not exist, the calculation is based on wav.scp)
    mv data/${train_set} data/${train_set}_org
    remove_longshortdata_onlybymaxframes.sh --maxframes 3000 data/${train_set}_org data/${train_set}
fi


# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Data augmentation"
  calc(){ awk "BEGIN { print "$*" }"; }
  frame_shift=`calc ${n_shift}/${fs}`
  echo "Calculated frame shift: ${frame_shift} sec"

  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/${train_set}/utt2num_frames > data/${train_set}/reco2dur

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
    data/${train_set} data/${train_set}_reverb
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/${train_set}_reverb data/${train_set}_reverb.new
  rm -rf data/${train_set}_reverb
  mv data/${train_set}_reverb.new data/${train_set}_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  if [ ! -f local/make_musan.sh ]; then
      kaldi_voxceleb_dir=/export/b13/jcho/espnet_v3/tools/kaldi/egs/voxceleb/v2/
      cp ${kaldi_voxceleb_dir}/local/make_musan.sh local/
      cp ${kaldi_voxceleb_dir}/local/make_musan.py local/

  fi
  local/make_musan.sh $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/${train_set} data/${train_set}_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/${train_set} data/${train_set}_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/${train_set} data/${train_set}_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/${train_set}_aug data/${train_set}_reverb data/${train_set}_noise data/${train_set}_music data/${train_set}_babble
fi

if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Feature extraction for 1M selected utts from data augmentated utts"
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh data/${train_set}_aug 1000000 data/${train_set}_aug_1m
  utils/fix_data_dir.sh data/${train_set}_aug_1m

  # Make fbanks for the augmented data.
  fbankdir=fbank
  for x in ${train_set}_aug_1m; do
    if [ ! -f ${x}/feats.scp ];then
      make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        data/${x} \
        exp/make_fbank/${x} \
        ${fbankdir}
    else
        echo "${x}/feats.scp already exists. Skip generating ${x}/feats.scp"
    fi
  done

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/${train_set}_combined data/${train_set}_aug_1m data/${train_set}
  utils/fix_data_dir.sh data/${train_set}_combined
fi

# (TODO (JJ): compare between having this stage and NOT: NOT having this stage but instead
# normalizing with one cmvn stat calculated from training has a priority)
# Now we prepare the features to generate examples for xvector training.
#if [ $stage -le 4 ]; then
#  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
#  # wasteful, as it roughly doubles the amount of training data on disk.  After
#  # creating training examples, this can be removed.
#  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
#    data/${train_set}_combined data/${train_set}_combined_no_sil exp/train_combined_no_sil
#  utils/fix_data_dir.sh data/${train_set}_combined_no_sil
#fi

# (TODO (JJ): compare between having this stage and NOT: NOT having this has a
# priority. By rough check, this does not filter out many (NOT AT ALL  from my check))
#if [ $stage -le 5 ]; then
#  # Now, we need to remove features that are too short after removing silence
#  # frames.  We want atleast 5s (500 frames) per utterance.
#  min_len=400
#  mv data/${train_set}_combined_no_sil/utt2num_frames data/${train_set}_combined_no_sil/utt2num_frames.bak
#  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/${train_set}_combined_no_sil/utt2num_frames.bak > data/${train_set}_combined_no_sil/utt2num_frames
#  utils/filter_scp.pl data/${train_set}_combined_no_sil/utt2num_frames data/${train_set}_combined_no_sil/utt2spk > data/${train_set}_combined_no_sil/utt2spk.new
#  mv data/${train_set}_combined_no_sil/utt2spk.new data/${train_set}_combined_no_sil/utt2spk
#  utils/fix_data_dir.sh data/${train_set}_combined_no_sil
#
#  # We also want several utterances per speaker. Now we'll throw out speakers
#  # with fewer than 8 utterances.
#  min_num_utts=8
#  awk '{print $1, NF-1}' data/${train_set}_combined_no_sil/spk2utt > data/${train_set}_combined_no_sil/spk2num
#  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/${train_set}_combined_no_sil/spk2num | utils/filter_scp.pl - data/${train_set}_combined_no_sil/spk2utt > data/${train_set}_combined_no_sil/spk2utt.new
#  mv data/${train_set}_combined_no_sil/spk2utt.new data/${train_set}_combined_no_sil/spk2utt
#  utils/spk2utt_to_utt2spk.pl data/${train_set}_combined_no_sil/spk2utt > data/${train_set}_combined_no_sil/utt2spk
#
#  utils/filter_scp.pl data/${train_set}_combined_no_sil/utt2spk data/${train_set}_combined_no_sil/utt2num_frames > data/${train_set}_combined_no_sil/utt2num_frames.new
#  mv data/${train_set}_combined_no_sil/utt2num_frames.new data/${train_set}_combined_no_sil/utt2num_frames
#
#  # Now we're ready to create training examples.
#  utils/fix_data_dir.sh data/${train_set}_combined_no_sil
#fi
# JJ add - end


# Stage to divide the data into train and dev subsets and generate utt2spklab for NN training is done
# at
# /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_voxcelebNaug_kaldiasr_phonealign_rf3/run.asrttsspkid.spkloss_weight.new.update.rf3.sh
# differently from other scripts for other corpora


# Check the utt2num_frames for data/${train_set}_combined to see if long utts
# exist (e.g. num_frames > 3000. Then, filter them out again) >> NOT exist even
# after data augmentation including RIR, (TODO(JJ): Check how RIR works under the hood)
# Refer to data/voxceleb_train_combined/feat2len.voxceleb_train_combined.txt
