#!/bin/bash

# JJ: Copied and edited from run.asrttsspkid.spkloss_weight.new.update.rf1.sh

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

# feature extraction related (set by following /export/b18/jcho/espnet3/egs/libritts/tts_featext/run.FE.diffconfig.voxceleb1.sh)
fs=16000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points (32ms) ***** JJ: maybe this leads to a poor result? (tacotron2 paper: 50ms, previous run.sh ~42ms)
n_shift=160   # number of shift points (10ms)
win_length="" # window length

# config files
train_config=conf/train_pytorch_tts+spkid_tacotron2+lresnet34.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Speaker id related
spkidloss_weight=0.03

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/60

# exp tag
tag="" # tag for managing experiments.

# data required to download ahead
voxceleb1_root="" # voxceleb1 corpus (should be manually downloaded)
voxceleb1_phoneali_fpath="data/files4reprod/phone.perframe.sym.txt" # phoneali file
uttlist_trainset="data/files4reprod/voxceleb1_train_filtered_train.uttlist" # uttlist_trainset
uttlist_devset="data/files4reprod/voxceleb1_train_filtered_dev.uttlist" # uttlist_devset
phone_dict="data/files4reprod/phones.txt" # phone dictionary
utt2spklab="data/files4reprod/voxceleb1_train_filtered.utt2spklab" # utt2spklab



. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

feat_dir=/export/b18/jcho/espnet3/egs/libritts/tts_featext
train_set=voxceleb1_train_filtered_train
dev_set=voxceleb1_train_filtered_dev
eval_set=voxceleb1_test

### jj add - start
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # 1. This script creates data/voxceleb1_test and data/voxceleb1_train.
    # Our evaluation set is the test portion of VoxCeleb1.
    local/make_voxceleb1.pl $voxceleb1_root data
    ## Download required files from gdrive. This generates data/files4reprod dir including the required files
    if [ ! -d data/files4reprod ]; then
        wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1vazEXMEbr3unk9PUTD969SZJp58zLXrX' -O files4reprod.tar.gz && tar -zxvf files4reprod.tar.gz -C data
    fi
    # 2. Phone alignment file ([uttid]-[phone alignment] pair in each line) preparation
    cp ${voxceleb1_phoneali_fpath} data/voxceleb1_train/text # # phone ali seq * 3 - (2, 3 or 4) = # acoustic seq
    ## remove the lines where text is empty and fix the directory
    mv data/voxceleb1_train/text data/voxceleb1_train/.text && awk '{if(!(NF==1))print $0}' data/voxceleb1_train/.text > data/voxceleb1_train/text # Extract only lines with non-empty phone alignment
    fix_data_dir.sh data/voxceleb1_train
    # 3. Separate into subsets: voxceleb1_train_filtered_train, voxceleb1_train_filtered_dev
    subset_data_dir.sh --utt-list ${uttlist_trainset} data/voxceleb1_train data/${train_set}
    subset_data_dir.sh --utt-list ${uttlist_devset} data/voxceleb1_train data/${dev_set}
    ls -A data/{${train_set},${dev_set}}/*
fi
### jj add - end

#if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#    echo "stage 0: Data preparation"
#    # 2. get subsets from /export/b18/jcho/espnet3/egs/libritts/tts_featext
#    mkdir data
#    for subset in ${train_set} ${dev_set} ${eval_set};do
#        cp -r ${feat_dir}/data/${subset} data/${subset}
#        ls -A data/${subset}
#    done
#    for subset in ${train_set} ${dev_set};do
#        cp /export/b11/jcho/kaldi_20190316/egs/librispeech/s5/exp/chain_cleaned/tdnn_1d_sp/decode_voxceleb1_all_frame_subsampling_factor_3/phone.perframe.sym.txt data/${subset}/text # # phone ali seq * 3 - (2, 3 or 4) = # acoustic seq
#        # remove the lines where text is empty and fix the directory
#        mv data/${subset}/text data/${subset}/.text && awk '{if(!(NF==1))print $0}' data/${subset}/.text > data/${subset}/text # JJ: This one does nothing in the kaldiali case
#        fix_data_dir.sh data/${subset}
#    done
#fi


feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    ### jj add - start
    # feature extraction with labrosa to generated data directories
    fbankdir=fbank
    for subset in ${train_set} ${dev_set} ${eval_set};do
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${subset} \
            exp/make_fbank/${subset} \
            ${fbankdir}
    done
    ### jj add - end
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
    cp ${phone_dict} ${dict}
    sed -e '1 s:<eps> 0:<unk> 0:' ${dict} | awk -F' ' '{$2=$2+1;print}' OFS=' ' > .tmpdict && mv .tmpdict ${dict}
    wc -l ${dict}

    # make json labels
    data2json_kaldiali.sh --kaldiali true --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json_kaldiali.sh --kaldiali true --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json

    # json prep. for speaker id module
    for subset in ${train_set} ${dev_set}; do
        # update json
        local/update_json_general.sh --loc output --k spklab ${dumpdir}/${subset}/data.json ${utt2spklab}
    done
fi

num_spk=`cat data/${train_set}/spk2utt | wc -l`
if [ ! $num_spk -eq 1211 ];then
    echo "num speaker: ${num_spk} is supposed to be 1211." && exit 1
fi
echo The number of speakers for spkid training: ${num_spk} # should be 1211 for voxceleb1

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_${tag}
fi

expname=${expname}_spkidloss_weight${spkidloss_weight}

expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --num-spk ${num_spk} \
           --spkidloss-weight ${spkidloss_weight} \
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

if [[ model.loss.best == ${model} ]]; then
    embname=`basename ${expdir}`
else
    embname=`basename ${expdir}`_${model}
fi

echo Embedding name: ${embname}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    pids=() # initialize pids
    # decode in parallel with cpus
    for extract_set in ${train_set} ${dev_set} ${eval_set}; do
    (
        # split feats.scp for cpu-parallel processing
        featscp=dump/${extract_set}/feats.scp
        splitdir=dump/${extract_set}/feats_split${nj}/
        if [ ! -d ${splitdir} ]; then
            mkdir -p ${splitdir}
            split_scps=""
            for n in $(seq $nj);do
                split_scps="${split_scps} ${splitdir}/feats.${n}.scp"
            done
            utils/split_scp.pl $featscp $split_scps || exit 1
        fi

        outemb=dump/${extract_set}/emb.${embname}
        ${train_cmd} JOB=1:${nj} log/speakerid_decode.${embname}.${extract_set}.JOB.log \
        speakerid_decode.py \
            --backend ${backend} \
            --ngpu 0 \
            --verbose ${verbose} \
            --out placeholder \
            --json placeholder \
            --model ${expdir}/results/${model} \
            --config ${decode_config} \
            --feat-scp ${splitdir}/feats.JOB.scp \
            --out-file ${outemb}.spltixJOB # if files with same names exist, files are overwritten from the beginning

        cat ${outemb}.spltix* | sort -k1 > ${outemb}
        rm ${outemb}.spltix*
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: backend-training"
    feat_trdt_dir=${dumpdir}/voxceleb1_train_filtered_trainNdev
    mkdir -p ${feat_trdt_dir}
    cat {${feat_tr_dir},${feat_dt_dir}}/emb.${embname} > ${feat_trdt_dir}/emb.${embname}
    # to combine utt2spk and spk2utt
    if [ ! -d data/voxceleb1_train_filtered_trainNdev ]; then
        combine_data.sh data/voxceleb1_train_filtered_trainNdev data/voxceleb1_train_filtered_train data/voxceleb1_train_filtered_dev
    fi

    $train_cmd ${feat_trdt_dir}/log/compute_emb_mean_${embname}.log \
        ivector-mean ark:${feat_trdt_dir}/emb.${embname} \
        ${feat_trdt_dir}/emb_mean.vec.${embname} || exit 1;

    # This script uses LDA to decrease the dimensionality prior to PLDA.
    lda_dim=150
    $train_cmd ${feat_trdt_dir}/log/lda.${embname}.log \
        ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
        "ark:ivector-subtract-global-mean ark:${feat_trdt_dir}/emb.${embname} ark:- |" \
        ark:data/voxceleb1_train_filtered_trainNdev/utt2spk ${feat_trdt_dir}/emb_transform.mat.${embname} || exit 1;

    # Train an out-of-domain PLDA model.
    $train_cmd ${feat_trdt_dir}/log/plda.${embname}.log \
        ivector-compute-plda ark:data/voxceleb1_train_filtered_trainNdev/spk2utt \
        "ark:ivector-subtract-global-mean ark:${feat_trdt_dir}/emb.${embname} ark:- | transform-vec ${feat_trdt_dir}/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
        ${feat_trdt_dir}/plda.${embname} || exit 1;
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "stage 7: Calculate score"
  # Get results using the out-of-domain PLDA model.
  $train_cmd exp/scores/log/spkid_scoring_${embname}.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${feat_trdt_dir}/plda.${embname} - |" \
    "ark:ivector-subtract-global-mean ${feat_trdt_dir}/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec ${feat_trdt_dir}/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${feat_trdt_dir}/emb_mean.vec.${embname} ark:dump/${eval_set}/emb.${embname} ark:- | transform-vec ${feat_trdt_dir}/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat data/${eval_set}/trials | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_${eval_set}_embname${embname} || exit 1;
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "stage 8: EER & MinDCF"
  qsub -cwd -o log/qsub.run.eer_mindcf.voxceleb1.${embname}.log -e log/qsub.run.eer_mindcf.voxceleb1.${embname}.log -l mem_free=200G,ram_free=200G run.eer_mindcf.voxceleb1.sh --eval_set ${eval_set} --embname ${embname}
fi
