# kaldi feat -> syn ?

. ./path.sh
## feature extraction related
fs=24000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
win_length="" # window length
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
#n_shift=240   # number of shift points
n_mels=80     # number of mel basis
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

. utils/parse_options.sh || exit 1;

## fbank extraction
compute-fbank-feats.py \
    --fs ${fs} \
    --fmax ${fmax} \
    --fmin ${fmin} \
    --win_length ${win_length} \
    --n_fft ${n_fft} \
    --n_shift ${n_shift} \
    --n_mels 80 \
    --write-num-frames=ark,t:utt2num_frames.${n_shift}.1 \
    --compress=true \
    --filetype mat \
    --normalize 16 \
    scp:temp_checking_feat/wav1.scp \
    ark,scp:feat1.${n_shift}.ark,feat1.${n_shift}.scp
## wav synthesis
    convert_fbank_to_wav.py \
        --fs ${fs} \
        --fmax ${fmax} \
        --fmin ${fmin} \
        --win_length ${win_length} \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --n_mels ${n_mels} \
        --iters ${griffin_lim_iters} \
        scp:feat1.${n_shift}.scp \
        .

## main (actuall run): 1st and 2nd no difference in synthesized wav.s
## I think it is so as long as compute-fbank-feats.py and convert_fbank_to_wav.py are configured consistently
## However, kaldi fbank is not working (I think some config should set properly) with convert_fbank_to_wav.py
### 1st n_shift 240
#bash list_run_temp_syn.sh --n_shift 240
#mv 1012_133424_000001_000000.wav 1012_133424_000001_000000.n_shift240.wav
### 2nd n_shift 256
#bash list_run_temp_syn.sh --n_shift 256
#mv 1012_133424_000001_000000.wav 1012_133424_000001_000000.n_shift256.wav
### 3rd kaldi fbank
#steps/make_fbank.sh --fbank-config conf/fbank_fs24k.conf --write_utt2num_frames true data/temp_kaldifbank/ data/temp_kaldifbank/log/ data/temp_kaldifbank/fbank/
#convert_fbank_to_wav.py --fs 24000 --fmax --fmin --win_length --n_fft 600 --n_shift 240 --n_mels 80 --iters ${griffin_lim_iters} scp:data/temp_kaldifbank/feats.scp data/temp_kaldifbank/
