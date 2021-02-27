MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi # again b{11,12,13,14} are broken. So as a stopgap (FROM ABOVE) temporarily symlink to a different directory (from /export/b13/jcho/espnet_v3/tools/kaldi to /export/b15/jcho/espnet_fork_cmpr/tools/kaldi by "$ ln -s /export/b15/jcho/espnet_fork_cmpr/tools/kaldi /export/b18/jcho/espnet3/tools/kaldi"). 
# possible multiple kaldi locations in different b machines: /export/b11/jcho/kaldi_20190316, /export/b13/jcho/espnet_v3/tools/kaldi, /export/b15/jcho/espnet_fork_cmpr/tools/kaldi

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1 # due to breakdown in b13, temporarily symlink to a different directory (from /export/b13/jcho/espnet_v3/tools/kaldi to /export/b15/jcho/espnet_fork_cmpr/tools/kaldi by "$ ln -s /export/b15/jcho/espnet_fork_cmpr/tools/kaldi /export/b18/jcho/espnet3/tools/kaldi")
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
if [ -e $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $MAIN_ROOT/tools/venv/bin/activate
fi
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
