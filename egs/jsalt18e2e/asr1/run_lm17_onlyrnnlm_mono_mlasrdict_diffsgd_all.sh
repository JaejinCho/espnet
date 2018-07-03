# this is except librispeech, and csj which we know already work
for lang in assamese tagalog swahili lao zulu;do
    bash run_lm17_onlyrnnlm_mono_mlasrdict_diffsgd.sh --lm_lang ${lang} --ngpu 1 2>&1 | tee log/run_lm17_onlyrnnlm_mono_mlasrdict_diffsgd.${lang}.log
done
