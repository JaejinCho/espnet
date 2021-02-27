#!/usr/bin/env python3

# Description: This script removes the text and token in each utterance stored in the data.json file
# Usage: python remove_textNtoken_indatajson.py [in json] [out new json]

import sys

fname_in = sys.argv[1]
fname_out = sys.argv[2]

fhandle_write = open(fname_out,'w',encoding='utf-8')
for line in open(fname_in, 'r', encoding='utf-8'):
    if (not '"text"' in line) and (not '"token"' in line) and (not '"tokenid"' in line):
        if "                    ]," in line:
            fhandle_write.write(line.replace(',',''))
        else:
            fhandle_write.write(line)


fhandle_write.close()
