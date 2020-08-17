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

Go to the experimental directory

```
cd egs/voxceleb1/spkidtts
```

Run experiment

```
$ bash run.sh --ngpu [# gpus to use] --spkidloss_weight [spkidloss_weight] --voxceleb1_root [voxceleb1 corpus dir] --voxceleb1_phoneali_fpath [(uttid)-(phn. seq.) pairs file] --uttlist_trainset [uttlist_trainset] --uttlist_devset [uttlist_devset] --phone_dict [phone dictionary] --utt2spklab [(uttid)-(spklab) pairs file] # Setting "--spkidloss_weight 0.03" is the same as M-TTS + SpkID loss w/ ASR Phn. Align. SR3 in Table 5 of the paper
```

## Citation

Learning Speaker Embedding from Text-to-Speech
