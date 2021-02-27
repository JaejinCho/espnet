. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

printthis=

. utils/parse_options.sh || exit 1;

(cat exp/Readme.txt && echo ${printthis}) 2>&1 | tee log/cat.log

