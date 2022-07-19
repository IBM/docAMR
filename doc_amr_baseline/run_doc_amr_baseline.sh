set -o pipefail
set -o errexit
. set_environment.sh
HELP="$0 <path_to_tokenized_sentence_amr> <path_to_out> <representation format no-merge -- No node merging, only chain-nodes
        merge-names -- Merge only names
        docAMR -- Merge names and drop pronouns
        merge-all -- Merge all nodes> <path_to_coref>"
[ -z $1 ] && echo "$HELP" && exit 1
[ -z $2 ] && echo "$HELP" && exit 1

path_to_sentence_amr=$1
out_amr=$2
rep=$3
path_to_coref=$4
set -o nounset

if [ -z $rep ];then
    rep='docAMR'
fi

if [ -z $path_to_coref ];then
    echo "Getting coref for sentences"
    coref_filename='allen_spanbert_large-2021.03.10.coref'
    python doc_amr_baseline/get_allen_coref.py \
        --path_to_sen $path_to_sentence_amr \
        --from_amr 
    path_to_coref=$path_to_sentence_amr/$coref_filename
fi

echo "Doc Level AMRs:"
python doc_amr_baseline/make_doc_amr.py \
    --path_to_coref $path_to_coref \
    --path_to_amr $path_to_sentence_amr \
    --out_amr $out_amr \
    --add_coref \
    --add_id \
    --allennlp \
    --norm_rep $rep \
    --sort_alpha






