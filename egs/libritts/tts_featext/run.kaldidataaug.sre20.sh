# To augment data in a kaldi way (upto 2 times of the original amount) and
# generate *_aug_oriamt directories finally. *_combined* directories will be
# generated in an experiment directory such as ../asrtts+spkid_sreNkaldiaug_phonealign

corpora_list_lab="fisher_spa sre16_eval_tr60_tgl sre16_eval_tr60_yue sre16_train_dev_ceb sre18_cmn2_train_lab sre_tel swbd voxcelebcat_tel sre16_train_dev_cmn"
corpora_list_nolab="babel_amharic babel_assamese babel_bengali babel_cantonese babel_cebuano babel_dholuo babel_georgian babel_guarani babel_haitian babel_igbo babel_javanese babel_kazakh babel_kurmanji babel_lao babel_lithuanian babel_mongolian babel_pashto babel_swahili babel_tagalog babel_tamil babel_telugu babel_tokpisin babel_turkish babel_vietnamese babel_zulu"

# kaldi data augmentation (this includes *** feature extraction in the end)
for cname in ${corpora_list_lab} ${corpora_list_nolab}; do
    bash dataaug_kaldi.sh --train_set ${cname} 2>&1 | tee log/dataaug_kaldi.${cname}.log
done
