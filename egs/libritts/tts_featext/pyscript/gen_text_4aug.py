# This script will generate text for augmented portion of data (py3)
# Output text will be written to [augmented dir]
# Usage: python gen_text_4aug.py [text file in ori dir b4 augmented] [augmented dir]
# e.g) python gen_text_4aug.py data/voxceleb_train/text  data/voxceleb_train_aug_1m/
import sys

fname_text = sys.argv[1]
dname_aug = sys.argv[2]
num_utt_aug = 1000000

utt2text = {}
for line in open(fname_text):
    utt, text = line.strip().split(' ',1)
    utt2text[utt] = text

fhandle_write = open(dname_aug + '/text', 'w')
cnt_ignored = 0
cnt_processed = 0
for line in open(dname_aug + '/utt2spk'):
    uttfull, spk = line.strip().split()
    utt = uttfull.strip().rsplit('-',1)[0]
    if utt in utt2text:
        fhandle_write.write(uttfull + ' ' + utt2text[utt] + '\n')
        cnt_processed = cnt_processed + 1
    else:
        print("{0} is not in ori uttlist".format(uttfull))
        cnt_ignored = cnt_ignored + 1

assert cnt_processed + cnt_ignored == num_utt_aug, 'cnt_processed + cnt_ignored is not {0}'.format(num_utt_aug)
fhandle_write.close()
print('# utt processed for text: {0}'.format(cnt_processed))
print('# utt ignored for text: {0}'.format(cnt_ignored))
