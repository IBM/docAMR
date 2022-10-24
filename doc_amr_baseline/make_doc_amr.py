import json
import os
import glob
from itertools import groupby
from operator import itemgetter
import copy
import collections
import pickle
import tqdm

import argparse
from baseline_io import (
    

    read_amr_add_sen_id,
    read_amr_str_add_sen_id,
    read_amr3_docid
    
)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from doc_amr import make_doc_amrs,connect_sen_amrs

from amr_constituents import get_subgraph_by_id,get_constituents_from_subgraph



def get_node_from_subgraph(subgraph,beg,end,doc_amr=None,verbose=False):
    candidate_nodes = []
    secondary_candidates =[]
    for head in subgraph['constituents']:
        if head['indices'][0]==beg and head['indices'][1]==end+1:
                name = head['head']
                return head['nid']
        elif head['indices'][0] in range(beg,end+2) and head['indices'][1] in range(beg,end+2):
            candidate_nodes.append(head)       
        elif head['head_position'] is not None:
            if max(head['head_position'])==end: #or end == head['head_position'][0]: This made other nodes wrong
                secondary_candidates.append(head)

    # al_node = get_node_from_alignment(doc_amr, beg, end)    

    if len(candidate_nodes) == 1:
        if verbose:
            print('approx alignment node')
        name = candidate_nodes[0]['head']
        return candidate_nodes[0]['nid']
    elif len(candidate_nodes)>1:
        if verbose:
            print('subset alignment mindepth node')
        mindepth = min(candidate_nodes, key=lambda x:x['depth'])
        name = mindepth['head']
        return mindepth['nid']
    elif len(secondary_candidates)>0:
        if verbose:
            print('end alignment mindepth node')
        mindepth = min(secondary_candidates, key=lambda x:x['depth'])
        name = mindepth['head']
        return mindepth['nid']


    return None


def construct_triples(doc_amrs,from_sen_id,from_node_id,sen_node_pairs,relation,verbose=False):
    triples = []
    for (full_sen_id,full_node_id) in sen_node_pairs:
        from_node = doc_amrs[full_sen_id].nvars[full_node_id]
        to_node = doc_amrs[from_sen_id].nvars[from_node_id]
        if from_node is not None and to_node is not None:   
            trip = (full_sen_id+'.'+from_node,relation,from_sen_id+'.'+to_node)
            triples.append(trip)
        else:
            if from_node is None and verbose:
                
                print(full_node_id ,' node id is not recognized in sentence ',full_sen_id)
            elif to_node is None and verbose:
                
                print(from_node_id ,' node id is not recognized in sentence ',from_sen_id)
    return triples    


def process_coref_conll(amrs,coref_chains,add_coref=True,verbose=False,save_triples=False,out=None,relation='same-as',coref_type='allennlp'):
    corefs = {}
    for doc_id,doc_amrs in tqdm.tqdm(amrs.items()):
        doc_triples = []
        doc_sids = list(doc_amrs.keys())
        sid_done =[]
        if add_coref and doc_id in coref_chains:
            #getting subgraph information of each amr
            subgraphs = {f_id: get_constituents_from_subgraph(doc_amrs[f_id]) for f_id in doc_amrs if doc_amrs[f_id].root is not None and len(doc_amrs[f_id].alignments)>0 }
            for ent in coref_chains[doc_id]:
                min_id = (None,None)
                sen_node_pairs = []
                for mention in ent:
                    if coref_type=='conll':
                        sid = mention[0]
                    elif coref_type=='allennlp':
                        sid = mention[0]+1
                    beg = mention[1]
                    end = mention[2]
                    sen_id = doc_id+'.'+str(sid)
                    if sen_id in subgraphs:
                        node_id = get_node_from_subgraph(subgraphs[sen_id], beg, end,doc_amrs[sen_id],verbose=verbose)
                    else:
                        node_id = None
                    if node_id is None:
                        if sen_id in sid_done:
                            if verbose:
                                print('maybe inter amr coref ,node not found')
                        else:
                            if verbose:
                                print('node not found')
                                print(mention)
                        continue
                    else:
                        sid_done.append(sen_id)
                        
                    if min_id[0] is None :
                        min_id = (sid,node_id)
                    elif sid < min_id[0]:
                        min_full_id = doc_id+'.'+str(min_id[0])
                        sen_node_pairs.append((min_full_id,min_id[1]))
                        min_id = (sid,node_id)
                    else:
                        sen_node_pairs.append((sen_id,node_id))
                
                min_full_id = doc_id+'.'+str(min_id[0])
                triples = construct_triples(doc_amrs,min_full_id,min_id[1],sen_node_pairs,relation)
                doc_triples.extend(triples)
                
        corefs[doc_id] = (doc_triples,doc_sids,doc_id)
        if save_triples:
            with open(args.out_amr.replace('.amr','.triples'),'wb') as f1:
                pickle.dump(corefs,f1)

    return corefs
                    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_coref',type=str,required=True)
    parser.add_argument('--path_to_amr',type=str,help='path to folder containing list of amr files per document',required=True)
    parser.add_argument('--out_amr',type=str,help='path to output',required=True)
    parser.add_argument('--add_id',action='store_true',help='add id to amr')
    parser.add_argument('--add_coref',action='store_true',default=False,help='add coref to doc amr')
    parser.add_argument('--allennlp',action='store_true',help='coref format pickled coref chains allen_nlp')
    parser.add_argument('--conll',action='store_true',help='coref format is conll')
    parser.add_argument('--event',action='store_true',help='perform event coref')
    parser.add_argument('--norm_rep',type=str,default='docAMR',help='''normalization format 
        "no-merge" -- No node merging, only chain-nodes
        "merge-names" -- Merge only names
        "docAMR" -- Merge names and drop pronouns
        "merge-all" -- Merge all nodes''')
    parser.add_argument('--verbose',action='store_true',help='Print types of nodes found')
    parser.add_argument('--save_triples',action='store_true',help='save triples as pickle')
    parser.add_argument('--tokenize',action='store_true',help='::tok not availabble in the parse')
    parser.add_argument('--sort_alpha',action='store_true',help='sort doc names alphabetically, default is False ie sorting numerically')
    parser.add_argument('--use_penman',action='store_true',default=True,help='use penman graph to construct doc amr (in the case of wiki)')
    parser.add_argument('--path_to_penman',type=str,default = None,help='optional path to folder to get penman amr')

    
    args = parser.parse_args()
    pat = '*'
    amrs = {}
    coref = {}
    mentions = {}
    entities = {}
    amrs_dict = {}
    event_clusters = {}
    events = {}
    sort_alpha = True
    amrs_penman = {}
    amrs_penman_dict = {}
    
    args.path_to_amr+='/'
    assert args.norm_rep in ['docAMR','no-merge','merge-names','merge-all'],'Norm represenation should be one of the following docAMR, no-merge, merge-names, merge-all'
    
    if not glob.glob(args.path_to_amr+pat+'.amr'):
        if not glob.glob(args.path_to_amr+pat+'.parse'):
            raise Exception("--path_to_amr folder does not contain .amr files or .parse files ")
        else:
            sort_alpha = True
            filepaths = glob.iglob(args.path_to_amr+pat+'.parse')
    else:
        filepaths = glob.iglob(args.path_to_amr+pat+'.amr')
    
    filepaths = list(filepaths)

    #FIXME sorting of sentence amrs based on filename,change to a universal sorting method
    if 'msamr_df' in filepaths[0].split('/')[-1]:
            sort_alpha = True
    if sort_alpha or args.sort_alpha:
        sorted_filepaths = sorted(filepaths,key=lambda t: t.split('/')[-1].split('.')[0])
    else:
        #sort doc_<num> numerically
        sorted_filepaths = sorted(filepaths,key=lambda t: int(t.split('/')[-1].split('.')[0].split('_')[-1]))
    #sorted(filepaths,key=lambda t: t.split('.')[0])
    # sorted_filepaths_dict = {'doc_'+str(idx): item.split('/')[-1].split('.')[0] for idx,item in enumerate(sorted_filepaths)}
    #sorted_filepaths_dict = {'doc_'+str(idx): item.split('/')[-1].split('.')[0] for item in filepaths}
    for filepath in sorted_filepaths:
        doc_id = filepath.split('/')[-1].split('.')[0]
        if args.add_id:
            amrs[doc_id] = read_amr_add_sen_id(filepath, doc_id,remove_id=args.add_id,tokenize=args.tokenize)
            if args.path_to_penman is not None:
                amrs_penman[doc_id] = read_amr_add_sen_id(args.path_to_penman+filepath.split('/')[-1], doc_id,remove_id=args.add_id,tokenize=args.tokenize,ibm_format=False)
            else:
                amrs_penman[doc_id] = read_amr_add_sen_id(filepath, doc_id,remove_id=args.add_id,tokenize=args.tokenize,ibm_format=False)
        else:
            d_amrs,doc_id = read_amr3_docid(filepath,ibm_format=True)
            amrs[doc_id] = d_amrs
            if args.path_to_penman is not None:
                amrs_penman[doc_id],doc_id = read_amr3_docid(args.path_to_penman+filepath.split('/')[-1],ibm_format=False)
            else:
                amrs_penman[doc_id],doc_id = read_amr3_docid(filepath,ibm_format=False)
        
        amrs_penman_dict.update(amrs_penman[doc_id])

        amrs_dict.update(amrs[doc_id])
    
    
    if args.allennlp:
        #Getting coref from allen-nlp Spanbert model
        coref_chains = {}
        out = pickle.load(open(args.path_to_coref,'rb'))
        for i,(doc_id,val) in enumerate(out.items()):

            coref_chains[doc_id] = val
        assert len(coref_chains)>0,"Coref file is empty"
        corefs = process_coref_conll(amrs,coref_chains,args.add_coref,verbose=args.verbose,save_triples=args.save_triples,coref_type='allennlp')
    elif args.conll:
        from corefconversion.conll_transform import read_file as conll_read_file
        from corefconversion.conll_transform import compute_chains as conll_compute_chains

        coref_chains = {}
        out = conll_read_file(args.path_to_coref)
        for n,(i,val) in enumerate(out.items()):
            
            docid_spl = i.split('); part ')
            doc_id = docid_spl[0].split('/')[-1]+'_'+str(int(docid_spl[1]))
            coref_chains[doc_id] = conll_compute_chains(val)
            assert len(coref_chains)>0,"Coref file is empty"
        corefs = process_coref_conll(amrs,coref_chains,save_triples=args.save_triples,out=args.out_amr,coref_type='conll')
        


    
    
    #FIXME sorting of sentence amrs based on filename,change to a universal sorting method
    if args.add_id and not sort_alpha and not args.sort_alpha:
        corefs = collections.OrderedDict(sorted(corefs.items(),key=lambda t: int(t[0].split('.')[0].split('_')[-1])))
    else:
        corefs = collections.OrderedDict(sorted(corefs.items(),key=lambda t: t[0].split('.')[0]))
    
    #use_penman is set to True by default , penman format is used to construct the final doc-amr
    if args.use_penman:
        out_doc_amrs = make_doc_amrs(corefs=corefs,amrs=amrs_penman_dict,chains=False)
    else:
        out_doc_amrs = make_doc_amrs(corefs=corefs,amrs=amrs_dict,chains=False)

    out_dir = args.out_amr.rsplit('/',1)[0]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    #with open(args.out_amr+'/'+doc_id+'_docamr_'+args.norm_rep+'.out', 'w') as fid:
    with open(args.out_amr+'/'+args.path_to_amr.split('/')[-1]+'docamr_'+args.norm_rep+'.out', 'w') as fid:
        
        for doc_id,amr in tqdm(out_doc_amrs.items(),'writing doc-amrs'):
            
                damr = copy.deepcopy(amr)
                connect_sen_amrs(damr)
                damr.make_chains_from_pairs()
                damr.normalize(args.norm_rep)
                damr_str = damr.__str__()
                
                
                fid.write(damr_str)
        fid.close()



        

    
    
    
    



