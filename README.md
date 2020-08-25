# Learning speaker embedding from Text-to-Speech

This is my first code sharing on Github. Any comments to improve this repo are welcome

## Getting Started

### Installation

Clone this repo and install ESPnet as below. If using a different version of ESPnet, the installation and codes also need to change accordingly.

```
cd [cloned repo]/tools
make KALDI=[kaldi path] PYTHON_VERSION=3.6 TH_VERSION=1.0.1 CUDA_VERSION=10.0 # Remove the "KALDI=[kaldi path]" part if kaldi is NOT installed yet
```

### Experiments

1. Download voxceleb1 corpus to [voxceleb1 corpus dir]. Running "$ ls [voxceleb1 corpus dir]" shows "vox1_meta.csv  voxceleb1_test.txt  voxceleb1_wav"

2. Go to the experimental directory

```
cd egs/voxceleb1/spkidtts
```

3. Run experiment

```
$ bash run.sh --ngpu [# gpus to use. Using multiple gpus is likely NOT working with the current codes] --spkidloss_weight [spkidloss weight] --voxceleb1_root [voxceleb1 corpus dir]
$ # e.g., bash run.sh --ngpu 1 --spkidloss_weight 0.03 --voxceleb1_root /export/corpora5/VoxCeleb1_v1 # Setting "--spkidloss_weight 0.03" is the same as M-TTS + SpkID loss w/ ASR Phn. Align. SR3 in Table 5 of the paper
```

## Citation

Learning Speaker Embedding from Text-to-Speech
