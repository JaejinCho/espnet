# Outline: This runs evaluation with a trained model on every voxceleb cleaned trial lists
# Pre-requisite: a trained backend pipeline, extracted embeddings for dump/voxceleb1_train_filtered_trainNdev

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=0
stop_stage=100
nj=70
expdir=""
model=model.loss.best # or give just a specific snapshot (without any path info. e.g. snapshot.ep.6)
verbose=0    # verbose option (if set > 1, get more log)
decode_config=conf/decode.yaml

# training data related
train_set=voxceleb1_train_filtered_train

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

feat_dir=/export/b18/jcho/espnet3/egs/libritts/tts_featext/
trial_dir=/export/b16/jcho/hyperion_trials/egs/voxceleb/v1/data/voxceleb1_test/
all_set=voxceleb1 # a name of the subset where things related to "all" utterances are located
emb_set=voxceleb1_notyet # a name of the subset where processed features (including embeddings) from the NOTYET-processed utterances will be saved

# Stage 1: This will run only once regardless of expdir
if [ ! -d dump/${emb_set} ]; then
    echo "No dir: dump/${emb_set}. Preparing the dir"
    # filter feats.scp file
    if [ ! -d data/${emb_set} ]; then
        echo "No dir: data/${emb_set}. Preparing the dir"
        mkdir -p data/${emb_set}
        filter_scp.pl --exclude <(cat dump/voxceleb1_train_filtered_{train,dev}/feats.scp) ${feat_dir}/data/${all_set}/feats.scp > data/${emb_set}/feats.scp
    fi
    # dump the filtered feats.scp file
    mkdir -p dump/${emb_set}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${emb_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${emb_set} dump/${emb_set}
fi


# Stage 2: From here until the end, all processes are depending on embname (i.e., expdir given)
if [[ model.loss.best == ${model} ]]; then
    embname=`basename ${expdir}`
else
    embname=`basename ${expdir}`_${model}
fi
echo Embedding name: ${embname}
# split feats.scp for cpu-parallel processing
featscp=dump/${emb_set}/feats.scp
splitdir=dump/${emb_set}/feats_split${nj}/
if [ ! -d ${splitdir} ]; then
    mkdir -p ${splitdir}
    split_scps=""
    for n in $(seq $nj);do
        split_scps="${split_scps} ${splitdir}/feats.${n}.scp"
    done
    utils/split_scp.pl $featscp $split_scps || exit 1
fi

outemb=dump/${emb_set}/emb.${embname}
${train_cmd} JOB=1:${nj} log/speakerid_decode.embset${emb_set}_embname${embname}.JOB.log \
speakerid_decode.py \
    --backend pytorch \
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

# combine already extracted embeddings & just extracted embeddings
mkdir -p dump/${all_set}
cat dump/voxceleb1_train_filtered_trainNdev/emb.${embname} dump/${emb_set}/emb.${embname} > dump/${all_set}/emb.${embname} # `wc -l dump/${all_set}/emb.${embname}` should be matched with `wc -l ${feat_dir}/data/${all_set}/feats.scp`
if [[ `wc -l dump/${all_set}/emb.${embname} | awk '{print $1}'` != `wc -l ${feat_dir}/data/${all_set}/feats.scp | awk '{print $1}'` ]]; then
    echo "# extracted embeddings are wrong" && exit 1
fi

# Stage 3: Filter embeddings by the corresponding trial list & Backend eval.
for trial_set in trials_o_clean trials_e_clean trials_h_clean; do
    mkdir -p dump/${trial_set}
    trial_list=${trial_dir}/${trial_set}
    awk '{print $1,$2}' ${trial_list} | tr ' ' '\n' | sort -u > ${trial_dir}/uttlist.${trial_set}
    filter_scp.pl ${trial_dir}/uttlist.${trial_set} dump/${all_set}/emb.${embname} > dump/${trial_set}/emb.${embname}

    # stage 7: Backend scoring
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
      echo "stage 7: Calculate score"
      # Get results using the out-of-domain PLDA model.
      $train_cmd exp/scores/log/spkid_scoring_trialset${trial_set}_embname${embname}.log \
        ivector-plda-scoring --normalize-length=true \
        "ivector-copy-plda --smoothing=0.0 dump/voxceleb1_train_filtered_trainNdev/plda.${embname} - |" \
        "ark:ivector-subtract-global-mean dump/voxceleb1_train_filtered_trainNdev/emb_mean.vec.${embname} ark:dump/${trial_set}/emb.${embname} ark:- | transform-vec dump/voxceleb1_train_filtered_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean dump/voxceleb1_train_filtered_trainNdev/emb_mean.vec.${embname} ark:dump/${trial_set}/emb.${embname} ark:- | transform-vec dump/voxceleb1_train_filtered_trainNdev/emb_transform.mat.${embname} ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat ${trial_list} | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_trialset${trial_set}_embname${embname} || exit 1;
    fi

    # stage 8: Backend eval.
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      echo "stage 8: EER & MinDCF"
      qsub -cwd -o log/qsub.run.eer_mindcf.voxceleb1.trialset${trial_set}_embname${embname}.log -e log/qsub.run.eer_mindcf.voxceleb1.trialset${trial_set}_embname${embname}.log -l mem_free=10G,ram_free=10G run.eer_mindcf.voxceleb1.flextrial.sh --trial_list ${trial_list} --embname ${embname}
    fi
done


