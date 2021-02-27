awk '{print $1" "NF-1}' [ori]/spk2utt | sort -h -k2 > [ori]/spk2num
awk '{if($2<=10){print $0}}' [ori]/spk2num > spklist.uttbelow10
num_uttbelow10=`wc -l uttlist.uttbelow10`
#num_trainutt=$(expr 2000 - ${num_uttbelow10})

awk -v l=${num_uttbelow10} 'NR<=l{print >> train.uttlist;}' 

shuf utt2
awk -v num_trainutt=$(expr 2000 - $num_uttbelow10)
