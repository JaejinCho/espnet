# Description: This script cats and deletes some segments and write a new segment file
# by the rules below (***** TESTING NOW *****):
# 1) cat: If the time gap between the segment and the next one < [time_gap] sec while
# combining does not happen anymore if it goes over [max_cat] sec
# 2) delete: segments where its length < [ignore_below] sec
# Usage: python seg2newseg.py segments new_segments [time_gap] [max_cat] [ignore_below]
# e.g., python seg2newseg.py segments new_segments 0.5 4 1

import sys

fhandle = open(sys.argv[1])
new_fhandle = open(sys.argv[2],'w')
time_gap = float(sys.argv[3]) # time gap between chunks (in sec)
max_cat = float(sys.argv[4]) # max length before stopping catting chunks (in sec)
ignore_below = float(sys.argv[5]) # length to delete the processed chunks (in sec)
print("Open: {0}. Write on: {1}".format(sys.argv[1],sys.argv[2]))

len_acc = 0 # accumulated length
prev_start = 0 # start time of the previous accumul. segment
prev_end = 0 # end time of the previous accumul. segment
write_flag = 0 # not written yet if 0

# (TODO:DONE) Think about first and last segment lines
# deal with first segment
line1 = fhandle.readline().split()
len_this = float(line1[3]) - float(line1[2])
if len_this >= max_cat: # case 1: write since len >= max_cat
    # (TODO:DONE) write (current start and end points) + len_acc init
    seg_id = line1[1] + '-' + str(int(float(line1[2])*100)).zfill(7) + '_' + str(int(float(line1[3])*100)).zfill(7)
    new_fhandle.write('{0} {1} {2} {3}\n'.format(seg_id,line1[1],line1[2],line1[3])) # format: seg_id utt_id start_time end_time
    len_acc = 0
    # below two lines seem NOT  necessary but for consistency
    prev_start = 0
    prev_end = 0
else: # case 2: accumul. since len < max_cat
    len_acc = len_this
    prev_start = float(line1[2])
    prev_end = float(line1[3])
prev_uttid = line1[1]

# main for loop
for line in fhandle:
    temp = line.split()
    if prev_uttid == temp[1]:
        if len_acc != 0:
            if float(temp[2]) - prev_end <= time_gap:
                if float(temp[3]) - prev_start >= max_cat:
                    # (TODO:DONE) write (prev_start and float(temp[3]))
                    seg_id = temp[1] + '-' + str(int(prev_start*100)).zfill(7) + '_' + str(int(float(temp[3])*100)).zfill(7)
                    new_fhandle.write('{0} {1} {2} {3}\n'.format(seg_id,temp[1],prev_start,temp[3])) # format: seg_id utt_id start_time end_time
                    len_acc = 0
                    # below two lines seem NOT  necessary but for consistency
                    prev_start = 0
                    prev_end = 0
                    write_flag = 1
                else:
                    prev_end = float(temp[3])
                    len_acc = prev_end - prev_start
                    write_flag = 0
            else: # NO accumul. write or ignore
                if prev_end - prev_start >= ignore_below: # if accumul. chunk len >= [ignore_below], write (otherwise just ignore)
                    # (TODO:DONE) write + updaate len_acc + prev_*
                    seg_id = temp[1] + '-' + str(int(prev_start*100)).zfill(7) + '_' + str(int(prev_end*100)).zfill(7)
                    new_fhandle.write('{0} {1} {2} {3}\n'.format(seg_id,temp[1],prev_start, prev_end)) # format: seg_id utt_id start_time end_time
                len_acc = float(temp[3]) - float(temp[2])
                prev_start = float(temp[2])
                prev_end = float(temp[3])
                write_flag = 0
        else: # time gap is ignored here.
            len_this = float(temp[3]) - float(temp[2])
            if len_this >= max_cat: # case 1: write since len >= max_cat
                # (TODO:DONE) write (current start and end points) + len_acc init
                seg_id = temp[1] + '-' + str(int(float(temp[2])*100)).zfill(7) + '_' + str(int(float(temp[3])*100)).zfill(7)
                new_fhandle.write('{0} {1} {2} {3}\n'.format(seg_id,temp[1],temp[2],temp[3])) # format: seg_id utt_id start_time end_time
                len_acc = 0
                # below two lines seem NOT  necessary but for consistency
                prev_start = 0
                prev_end = 0
                write_flag = 1
            else: # case 2: accumul. since len < max_cat
                len_acc = len_this
                prev_start = float(temp[2])
                prev_end = float(temp[3])
                write_flag = 0
    else: # uttid change
        # finish the previous uttid
        if write_flag == 0:
            if prev_end - prev_start >= ignore_below: # if accumul. chunk len >= ignore_below, write (otherwise just ignore)
                # (TODO:DONE) write + updaate len_acc + prev_*
                seg_id = prev_uttid + '-' + str(int(prev_start*100)).zfill(7) + '_' + str(int(prev_end*100)).zfill(7)
                new_fhandle.write('{0} {1} {2} {3}\n'.format(seg_id,prev_uttid,prev_start, prev_end)) # format: seg_id utt_id start_time end_time
                write_flag = 1
        # process for the current uttid
        len_this = float(temp[3]) - float(temp[2])
        if len_this >= max_cat: # case 1: write since len >= max_cat
            # (TODO:DONE) write (current start and end points) + len_acc init
            seg_id = temp[1] + '-' + str(int(float(temp[2])*100)).zfill(7) + '_' + str(int(float(temp[3])*100)).zfill(7)
            new_fhandle.write('{0} {1} {2} {3}\n'.format(seg_id,temp[1],temp[2],temp[3])) # format: seg_id utt_id start_time end_time
            len_acc = 0
            # below two lines seem NOT  necessary but for consistency
            prev_start = 0
            prev_end = 0
            write_flag = 1
        else: # case 2: accumul. since len < max_cat
            len_acc = len_this
            prev_start = float(temp[2])
            prev_end = float(temp[3])
            write_flag = 0
    prev_uttid = temp[1]
else: # this is when I get to the eof.
    # finish the previous uttid
    if write_flag == 0:
        if prev_end - prev_start >= ignore_below: # if accumul. chunk len >= ignore_below, write (otherwise just ignore)
            # (TODO:DONE) write + updaate len_acc + prev_*
            seg_id = prev_uttid + '-' + str(int(prev_start*100)).zfill(7) + '_' + str(int(prev_end*100)).zfill(7)
            new_fhandle.write('{0} {1} {2} {3}\n'.format(seg_id,prev_uttid,prev_start, prev_end)) # format: seg_id utt_id start_time end_time
            write_flag = 1

new_fhandle.close()
