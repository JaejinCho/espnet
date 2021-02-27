cname=babel_assamese
dumpdir=dump
qsub -wd /export/b18/jcho/espnet3/egs/libritts/asrtts+spkid_sre_phonealign -o log/update_json_general.qsub.${cname}_nolab.log -e log/update_json_general.qsub.${cname}_nolab.log local/update_json_general.sh --loc output --k spklab ${dumpdir}/${cname}_nolab/data.json ./data/${cname}/utt2spklab
