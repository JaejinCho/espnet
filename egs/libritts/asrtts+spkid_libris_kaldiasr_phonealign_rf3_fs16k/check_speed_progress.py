# This script will check the speed progress to see where the process got slower
# Usage: python thisscript.py [train.log]

import sys

lines = open(sys.argv[1]).readlines()
ix_inc = 5 # this is a consistent line index change to get to the next time mark

for ix, line in enumerate(lines):
    if line.strip().split()[1] == 100:
        start_ix = ix
        num_iter = 100
        break

# save (num_iter, elapsed time) in each line being examined ***STOPPED DUE TO
# TOO SLOW GRID ***
