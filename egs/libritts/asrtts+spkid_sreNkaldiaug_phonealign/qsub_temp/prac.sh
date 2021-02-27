a=j1
jlist=""

for i in `seq 5`;do
    jlist+=",j${i}"
    if [ $i == 1 ]; then
        qsub -N j${i} -cwd -o qsub${i}.log -e qsub${i}.log qsub.sh
    else
        qsub -hold_jid j$((i-1)) -N j${i} -cwd -o qsub${i}.log -e qsub${i}.log qsub.sh
    fi
done

qsub -cwd -hold_jid ${jlist#,} -o final.log -e final.log final.sh


