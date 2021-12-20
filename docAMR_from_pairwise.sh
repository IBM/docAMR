
amr=$1
rep=$2
rel="same-as"

python doc_amr.py \
       --in-doc-amr-pairwise $amr \
       --pairwise-coref-rel $rel \
       --rep $rep \
       --out-amr $amr.$rep.out
