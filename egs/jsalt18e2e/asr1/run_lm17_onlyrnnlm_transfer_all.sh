# this is except assamese that is already run, librispeech, and csj which we
# know already work
for lang in assamese tagalog swahili lao zulu;do
    bash run_lm17_onlyrnnlm_transfer.sh --stage 3 --lm_lang ${lang} --ngpu 1 2>&1 | tee log/run_lm17_onlyrnnlm_transfer.${lang}.log &
done
