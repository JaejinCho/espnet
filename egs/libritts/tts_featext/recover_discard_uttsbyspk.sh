#!/bin/bash

test_data_dir=$1
seg_dir=$2
cp ${seg_dir}/segments ${seg_dir}/.segments_ori
python recover_discard_uttsbyspk.py ${test_data_dir}/utt2num_frames ${seg_dir}/segments

