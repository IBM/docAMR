set -o errexit 
set -o pipefail
set -o nounset 

[ -z $1 ] && echo "$0 <gold-docamr> <path-to-tokenized-sentence-amrs> <normalization-representation>" && exit 1
[ -z $2 ] && echo "$0 <gold-docamr> <path-to-tokenized-sentence-amrs> <normalization-representation>" && exit 1
[ -z $3 ] && echo "$0 <gold-docamr> <path-to-tokenized-sentence-amrs> <normalization-representation>" && exit 1

gold_amr=$1
sentence_amr=$2
rep=$3

# running baseline with AMR3 document amrs test split
bash run_doc_amr_baseline.sh $sentence_amr $sentence_amr $rep

echo "Computing Smatch "
python docSmatch/smatch.py -r 10 --significant 4 -f $gold_amr ${sentence_amr}/docamr_${rep}.out

printf "[\033[92m OK \033[0m] $0\n"