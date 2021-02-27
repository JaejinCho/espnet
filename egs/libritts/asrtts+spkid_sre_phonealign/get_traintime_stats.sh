#!/bin/bash

# Description: This script is to generate a gpu node index & statictics (min,max,mean,std) for the time spent (per 100 iteration) in training
# Usage: bash get_traintime_stats.sh [train.log file]
log_file=$1

head -n1 ${log_file}

grep -A1 "Estimated time to finish" ${log_file} | grep -v "Estimated time to finish" | awk '{print $3}' | xargs > tmp.input
python get_traintime_stats.py tmp.input


