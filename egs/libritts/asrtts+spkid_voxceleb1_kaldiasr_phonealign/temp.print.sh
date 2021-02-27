. path.sh
. cmd.sh
arg1=""
arg2=""
. utils/parse_options.sh || exit 1;

var1=$1
var2=$2
echo $arg1 $arg2 $var1 $var2 2>&1 | tee temp.print.log
