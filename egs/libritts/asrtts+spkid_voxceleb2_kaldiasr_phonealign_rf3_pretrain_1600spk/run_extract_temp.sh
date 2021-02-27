#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0
stop_stage=100
ngpu=1       # number of gpu in training
nj=64        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=24000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# config files
train_config=conf/train_pytorch_tacotron2+spkemb.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Speaker id related
spkloss_weight=0
extract_set=

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

train_set_ori=../tts1/data/train_clean_460 # JJ: If train_clean_460_org is used (long duration utts are not filtered), CUDA memory error seems to happen
train_set=train_train
dev_set=train_dev
eval_set=train_eval

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    if [ ! -d ${train_set_ori} ]; then
        echo "Generate ${train_set_ori} directory first" && exit 1
    fi
    mkdir -p data && cp -r ${train_set_ori} data
    # 1. get uttlists for new subset division (train_train, train_dev,
    # train_eval) + train_trainNdev.utt2spklab for spk discriminative training
    python make_uttlists.py data/train_clean_460/
    # 2. get subsets by uttlist files
    for fpath in `ls data/train_clean_460/*uttlist`;do
        dname=$(basename ${fpath} | cut -d'.' -f1)
        subset_data_dir.sh --utt-list ${fpath} data/train_clean_460/ data/${dname}/
        ls -A data/${dname}/*
    done
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/eval ${feat_ev_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json

    # json prep. for speaker id module
    for subset in ${train_set} ${dev_set}; do
        # update json
        local/update_json_general.sh --loc output --k spklab ${dumpdir}/${subset}/data.json data/train_clean_460/train_trainNdev.utt2spklab # utt2spklab generated same way as for utt2spkid from Nanxin's Resnet code
    done
fi
num_spk=`cat data/${train_set}/spk2utt | cut -d'_' -f1 | sort -u | wc -l` # should be 1k for now
if [ ! $num_spk -eq 1000 ];then
    echo "num speaker: ${num_spk} is supposed to be 1k." && exit 1
fi

echo The number of speakers for spkid training: ${num_spk}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi

expname=${expname}_spkloss_weight${spkloss_weight}

expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train_speakerid.py \
           --model-module espnet.nets.pytorch_backend.e2e_tts_tacotron2_speakerid:Tacotron2 \
           --num-spk ${num_spk} \
           --train-spkid-extractor True \
           --train-spk-embed-dim 400 \
           --spkloss-weight ${spkloss_weight} \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

## JJ - start
#if [ ${n_average} -gt 0 ]; then
#    model=model.last${n_average}.avg.best
#fi
model=exp/train_train_pytorch_train_pytorch_tacotron2+spkemb_speakeridonly_ttslossw0_spklossw1/results/model.loss.best
featscp=dump/${extract_set}/feats.scp
outemb=dump/${extract_set}/emb.temp.txt.ark
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
        # decode in parallel
        ${cuda_cmd} --gpu ${ngpu} log/speakerid_decode.${embname}.log \
            speakerid_decode.py \
                --backend ${backend} \
                --ngpu 1 \
                --verbose ${verbose} \
                --out placeholder \
                --json placeholder \
                --model ${model} \
                --config ${decode_config} \
                --feat-scp ${featscp} \
                --out-file ${outemb}
fi

# stage 8

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: backend-training"
  mkdir -p dump/train_trainNdev && cat dump/train_{train,dev}/emb.temp.txt.ark > dump/train_trainNdev/emb.temp.txt.ark
  combine_data.sh data/train_trainNdev data/train_train data/train_dev

  $train_cmd dump/train_trainNdev/log/compute_emb_mean.log \
      ivector-mean ark:dump/train_trainNdev/emb.temp.txt.ark \
      dump/train_trainNdev/emb_mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd dump/train_trainNdev/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean ark:dump/train_trainNdev/emb.temp.txt.ark ark:- |" \
    ark:data/train_trainNdev/utt2spk dump/train_trainNdev/emb_transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd dump/train_trainNdev/log/plda.log \
    ivector-compute-plda ark:data/train_trainNdev/spk2utt \
    "ark:ivector-subtract-global-mean ark:dump/train_trainNdev/emb.temp.txt.ark ark:- | transform-vec dump/train_trainNdev/emb_transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    dump/train_trainNdev/plda || exit 1;
fi
# stage 9
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 7: Calculate score"
  # Get results using the out-of-domain PLDA model.
  $train_cmd exp/scores/log/spkid_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 dump/train_trainNdev/plda - |" \
    "ark:ivector-subtract-global-mean dump/train_trainNdev/emb_mean.vec ark:dump/train_eval/emb.temp.txt.ark ark:- | transform-vec dump/train_trainNdev/emb_transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean dump/train_trainNdev/emb_mean.vec ark:dump/train_eval/emb.temp.txt.ark ark:- | transform-vec dump/train_trainNdev/emb_transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat data/train_eval/trials | cut -d\  --fields=1,2 |" exp/scores_libritts_train_eval || exit 1;
fi
# stage 10
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "stage 8: EER & MinDCF"
  #eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  #mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  #mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  #eer=`compute-eer <(local/prepare_for_eer.py data/train_eval/trials exp/scores_libritts_train_eval) 2> /dev/null`
  eer=$(paste data/train_eval/trials exp/scores_libritts_train_eval | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_libritts_train_eval data/train_eval/trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_libritts_train_eval data/train_eval/trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
### JJ - end
