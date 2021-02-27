# 1. get uttlists for new subset division (train_train, train_dev, train_eval)
# ../tts1/data/train_clean_460_org has 1151 (also same as in the LibriTTS paper)
# 1k spks for train_train, the rest 151 spks for train_{dev,eval}
## 1. build a actual spk2utt file (Ones used in tts1/tts+spkid directories were session2utt files)
mkdir -p data && cp -r ../tts1/data/train_clean_460_org data
python make_uttlists.py data/train_clean_460_org/

# 2. get subsets by uttlist files
for fpath in `ls data/train_clean_460_org/*uttlist`;do
    dname=$(basename ${fpath} | cut -d'.' -f1)
    subset_data_dir.sh --utt-list ${fpath} data/train_clean_460_org/ data/${dname}/
    ls -A data/${dname}/*
done
