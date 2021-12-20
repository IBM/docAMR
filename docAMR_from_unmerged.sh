


amr=$1
rep=$2

python doc_amr.py \
       --in-doc-amr-unmerged $amr \
       --rep $rep \
       --out-amr $amr.$rep.out

