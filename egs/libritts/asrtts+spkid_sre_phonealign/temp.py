import random
import argparse

parser = argparse.ArgumentParser(description='Average segment-wise x-vectors')
parser.add_argument('fname_in_emb', type=str,
                    help='input: segment-wise embdding file name')
parser.add_argument('fname_out_emb', type=str,
                    help='output: averaged utt-wise embedding file name')
parser.add_argument('--random', type=str, dest='fname_in_numframes', default=None,
                    help='utt2num_frames file name if we want to do random segments selection before average. Currently total # frames from the random segments ranges between 1k and 6k in frames')
args = parser.parse_args()

print(args.fname_in_emb)
print(args.fname_out_emb)
if args.fname_in_numframes:
    print(args.fname_in_numframes)
