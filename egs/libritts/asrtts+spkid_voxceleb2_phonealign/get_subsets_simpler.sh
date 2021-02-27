# This will compose data subest directories roughly with 1x, 2x, 3x, ...
# utteracnes of # utts in voxceleb1 (dir with higher multiples include the
# speakers in the lower multiples)

datadir=data/voxceleb2
#vox1train_num_spk=1211

if [ ! -f ${datadir}/spklist ]; then
    awk -F' ' '{print $1}' ${datadir}/spk2utt | sort -u | shuf > ${datadir}/spklist
    echo "# speakers: $(wc -l ${datadir}/spklist)" # expected to be 6114 for data/voxceleb2
fi

# 800 spk
subset_data_dir.sh --spk-list <(head -n800 ${datadir}/spklist) ${datadir} ${datadir}_800spk_stack
# 1600 spk
subset_data_dir.sh --spk-list <(head -n1600 ${datadir}/spklist) ${datadir} ${datadir}_1600spk_stack
# 2400 spk
subset_data_dir.sh --spk-list <(head -n2400 ${datadir}/spklist) ${datadir} ${datadir}_2400spk_stack
