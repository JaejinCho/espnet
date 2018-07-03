import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=str,required=True)

args = parser.parse_args()

print(args.lr)
