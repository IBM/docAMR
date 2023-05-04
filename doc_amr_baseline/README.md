## Run DocAMR Baseline
To setup the environment we use a file called set_environment.sh

```bash
touch set_environment.sh
```

The activation of the conda/virtual environment can be added inside this file

Packages to install are in requirements.txt. Python 3.7 works best. To install the packages required, perform the following inside the conda/virtual environment.

```bash
pip install -r doc_amr_baseline/requirements.txt
```

To get a document amr , given the sentence amrs run

```bash
bash doc_amr_baseline/run_doc_amr_baseline.sh <path_to_tokenized_sentence_amrs> <path_to_out> <normalization_representation> <path_to_coref-optional>

```
<path_to_tokenized_sentence_amrs> is a folder containing a file of sentence amrs for each document . Each file in the folder should have extension '.amr' and contain sentence amrs for all sentences in the document seperated by a newline. See **Format of AMR files** for further details.

<path_to_out> folder the doc amr for each document is to be output

<normalization_representation>  

    "no-merge" -- No node merging, only chain-nodes
    "merge-names" -- Merge only names
    "docAMR" -- Merge names and drop pronouns
    "merge-all" -- Merge all nodes

Recommended representation based on the [paper](https://aclanthology.org/2022.naacl-main.256.pdf) is **"docAMR"**

<path_to_coref> path to allennlp coref in pickled format, is optional (ie previously generated coref can be reused here). If not provided, the script will use the sentences in the amrs to get spanBert coref from "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"

## Format of AMR files expected for DocAMR baseline
Each file inside the folder <path_to_tokenized_sentence_amrs> should
1. End with extension '.amr'
2. Contain sentence amrs for all sentences in the document seperated by a newline
3. Contain metadata information about the AMR parse such as alignments and node id. 

See example folder for sample .amr file



## Run DocAMR Baseline test

To run a test of the baseline, given the gold docamr and sentence amrs ,

```bash
bash doc_amr_baseline/tests/baseline_allennlp_test.sh <gold-docamr> <path-to-tokenized-sentence-amrs> <normalization-representation>

```

<gold-docamr> is a file containing the gold docamr obtained using the command mentioned in the main README with the same representation as <normalization-representation>

```bash
python doc_amr.py 
--amr3-path <path to AMR3 data> 
--coref-fof <file-with-list-of-xml-annotations-files> 
--out-amr <output file> 
--rep <representation>

```

<path_to_tokenized_sentence_amrs> is a folder containing a file of sentence amrs for each document . Each file in the folder should have extension '.amr' and contain sentence amrs for all sentences in the document seperated by a newline. See **Format of AMR files** for further details.

<normalization_representation>  

    "no-merge" -- No node merging, only chain-nodes
    "merge-names" -- Merge only names
    "docAMR" -- Merge names and drop pronouns
    "merge-all" -- Merge all nodes

Recommended representation based on the [paper](https://aclanthology.org/2022.naacl-main.256.pdf) is **"docAMR"**



