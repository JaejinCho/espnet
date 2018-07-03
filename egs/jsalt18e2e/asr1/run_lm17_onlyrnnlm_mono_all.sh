# this is except assamese that is already run, librispeech, and csj which we
# know already work
for lang in `sed -n '2,15p' lang_list`;do
    bash run_lm17_onlyrnnlm_mono.sh --lm_lang ${lang} --ngpu 1 2>&1 | tee log/run_lm17_onlyrnnlm_mono.${lang}.log
done
