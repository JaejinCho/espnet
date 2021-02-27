dname_log=$1
fname_log=`echo $dname_log`/results/log
for ltype in `grep validation ${fname_log} | awk -F'"' '{print $2}'| sort -u`;do
    echo ${ltype}
    grep -w ${ltype} ${fname_log} | cat -n | sort -rn -k3
done
