#conda activate allen_nlp
import allennlp
from allennlp.predictors.predictor import Predictor
from itertools import accumulate
# import allennlp_models.tagging
import glob
import pickle
from tqdm import tqdm
import argparse

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

def get_allen_coref(filepath,from_amr=False):
    
    f1 = open(filepath,'r').read()
    sen_list = f1.splitlines()
    if from_amr:
        sen_tok_list = [s.split('::tok ')[-1].split() for s in sen_list if '::tok' in s]
    else:
        sen_tok_list = [s.split() for s in sen_list]
    sen_tok_len = [len(l) for l in sen_tok_list]
    sen_tok_len = list(accumulate(sen_tok_len))
    sen_tok = [item for sublist in sen_tok_list for item in sublist]


    pred = predictor.predict_tokenized(tokenized_document=sen_tok)
    clusters = pred['clusters']
    document = pred['document']
    new_cluster = []

    for c in clusters:
        new_c = []
        for m in c:
            for idx,l in enumerate(sen_tok_len):
                if m[0]< l:
                    prev_len = sen_tok_len[idx-1]
                    break
            if idx!=0:
                new_c.append([idx,m[0]-prev_len,m[1]-prev_len])
            else:
                new_c.append([idx,m[0],m[1]])
        new_cluster.append(new_c)
    return new_cluster        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_sen',type=str)
    parser.add_argument('--path_to_out',type=str,help='path to output',default=None)
    parser.add_argument('--from_amr',action='store_true')
    parser.add_argument('--from_json',action='store_true')

    args = parser.parse_args()
    doc_clusters = {}
    args.path_to_sen+='/'
   
    
    i = 0
    if args.from_amr:
        ext = '.amr'
    else:
        ext = '.txt'
    
    # if from_json:
    #     json_dict = json.load(open(path_to_sen))
    #     for doc_id,doc_val in json_dict.items():
    #             amr_strs = [s['sentence'] for s in doc_val['sentences'].values()]
    # path_fill = path_to_sen+'doc*'+ext
    path_fill = args.path_to_sen+'*'+ext

    for filepath in tqdm(glob.iglob(path_fill)):
        doc_id = filepath.split('/')[-1].split('.')[0]
        clusters = get_allen_coref(filepath,from_amr=args.from_amr)
        doc_clusters[doc_id] = clusters
        i+=1
    
    if args.path_to_out is None:
        out_path = args.path_to_sen+'/allen_spanbert_large-2021.03.10.coref'
    else:
        out_path = args.path_to_out+'/allen_spanbert_large-2021.03.10.coref'
    with open(out_path,'wb') as f2:
        pickle.dump(doc_clusters,f2)
