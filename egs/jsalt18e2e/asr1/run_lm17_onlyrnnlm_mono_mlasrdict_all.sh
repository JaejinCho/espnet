# this is except librispeech, and csj which we know already work
for lang in `sed -n '1,15p' lang_list`;do
    bash run_lm17_onlyrnnlm_mono_mlasrdict.sh --lm_lang ${lang} --ngpu 1 2>&1 | tee log/run_lm17_onlyrnnlm_mono_mlasrdict.${lang}.log
done
