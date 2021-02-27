# Description: Was written quickly to check if the trials for SV were generated correctly
# Usage: python thisscript.py
# Compare the output (trials.temp) from this script with data/voxceleb1_test/trials
fin = open('/export/corpora/VoxCeleb1/voxceleb1_test.txt') # Jesus checked me if I generate my trials from this file
fout = open('./trials.temp','w')

for line in fin:
    a,b,c = line.split()
    b_spk, b_utt = b.split('/')
    c_spk, c_utt = c.split('/')
    if a == '1':
        fout.write(b_spk + '-' + b_utt.split('.wav')[0][::-1].replace('_','-',1)[::-1] + ' ' + c_spk + '-' + c_utt.split('.wav')[0][::-1].replace('_','-',1)[::-1] + ' target\n')
    elif a == '0':
        fout.write(b_spk + '-' + b_utt.split('.wav')[0][::-1].replace('_','-',1)[::-1] + ' ' + c_spk + '-' + c_utt.split('.wav')[0][::-1].replace('_','-',1)[::-1] + ' nontarget\n')
    else:
        print('wrongwrongwrong')

fout.close()
