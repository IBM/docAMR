

amr3path="/dccstor/ykt-parse/SHARED/CORPORA/AMR/amr_annotation_3.0"
split=$1
rep=$2
output_dir="outputs"

mkdir -p $output_dir

python doc_amr.py \
       --amr3-path $amr3path \
       --coref-fof ${split}_coref.fof \
       --rep $rep \
       --out-amr $output_dir/${split}.gold.$rep.out

