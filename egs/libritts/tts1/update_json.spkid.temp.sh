train_set=train_clean_460
dev_set=dev_clean
eval_set=test_clean

#for subset in ${train_set} ${dev_set} ${eval_set}; do
#    # generate uut2spklab
#    awk 'BEGIN{s=0;}{if(!($2 in a)){a[$2]=s;s+=1;}print $1,a[$2]}' data/${subset}/utt2spk > data/${subset}/utt2spklab
#    mkdir -p dump_spkidadd/${subset} && cp dump/${subset}/.backup/data.json dump_spkidadd/${subset}/data.json
#    local/update_json_general.sh --loc output -k spklab dump_spkidadd/${subset}/data.json data/${subset}/utt2spklab
#done

# Step 1: Divde train_clean_460 into two parts (train (0.9) and cv (0.1)).
# This needs since we have to include all the speakers from cv in train.

## set vars
train_ori=train_clean_460
train_spktrain=train_clean_460_spktrain_0.9
train_spkcv=train_clean_460_spkcv_0.1

## cnt utts in each spk
awk '!($2 in a){a[$2]=0} {a[$2]+=1} END{for(i in a) print i, a[i]}' data/${train_ori}/utt2spk > data/${train_ori}/spk2num

## divide to train and cv (put into train all the utterances from a speaker where its #utts is less than 10)
### generate uttlist
awk -v data_dir=data/${train_ori}/ 'BEGIN{srand(1);} FNR==NR{a[$1]=$2; next} {if(a[$2]<10)print $1 >> data_dir"train.spkid.uttlist";else{if(rand()<=0.1)print $1 >> data_dir"cv.spkid.uttlist";else print $1 >> data_dir"train.spkid.uttlist"}}' data/${train_ori}/spk2num data/${train_ori}/utt2spk
### Divide data/${train_ori}/. NOT divide dump/${train_ori}/feats.scp (since cmvn.ark is calculated again)
utils/subset_data_dir.sh --utt-list data/${train_ori}/train.spkid.uttlist data/${train_ori} data/${train_spktrain}
utils/subset_data_dir.sh --utt-list data/${train_ori}/cv.spkid.uttlist data/${train_ori} data/${train_spkcv}
#From the above, these are not extracted: cmvn.ark cv.spkid.uttlist spk2num tmp train.spkid.uttlist utt2spklab
#though no need to deal with them separately for division

## CMVN
