from tqdm import tqdm
import penman
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from amr_io import AMR
import re
from collections import defaultdict

alignment_regex = re.compile('(-?[0-9]+)-(-?[0-9]+)')
class AMR2(AMR):
    @classmethod
    def fix_alignments(self,removed_idx):
        for node_id,align in self.alignments.items():
            if align is not None:
                num_decr = len([rm for rm in removed_idx if align[0]>rm])
                if num_decr>0:
                    lst = [x-num_decr for x in align]
                    self.alignments[node_id] = lst 


    @classmethod
    def get_sen_ends(self):
        self.sentence_ends = []
        removed_idx = []
        new_tokens = []
        for idx,tok in enumerate(self.tokens):
            if tok=='<next_sent>':
                self.sentence_ends.append(len(new_tokens)-1)
                removed_idx.append(idx)
            else:
                new_tokens.append(tok)
        self.sentence_ends.append(len(new_tokens)-1)
        self.tokens = new_tokens
        self.fix_alignments(removed_idx)
        
    @classmethod
    #FIXME
    def remove_unicode(self):
        for idx,tok in enumerate(self.tokens):
            new_tok = tok.encode("ascii", "ignore")
            self.tokens[idx] = new_tok.decode()
            if self.tokens[idx]=='':
                self.tokens[idx]='.'

    @classmethod
    def get_all_vars(cls, penman_str):
        in_quotes = False
        all_vars = []
        for (i,ch) in enumerate(penman_str):
            if ch == '"':
                if in_quotes:
                    in_quotes = False
                else:
                    in_quotes = True
            if in_quotes:
                continue
            if ch == '(':
                var = ''
                j = i+1
                while j < len(penman_str) and penman_str[j] not in [' ','\n']:
                    var += penman_str[j]
                    j += 1
                all_vars.append(var)
        return all_vars
    
    @classmethod
    def get_node_var(cls, penman_str, node_id):
        """
        find node variable based on ids like 0.0.1
        """
        nid = '99990.0.0.0.0.0.0'
        cidx = []
        lvls = []
        all_vars = AMR2.get_all_vars(penman_str)
        in_quotes = False
        for (i,ch) in enumerate(penman_str):
            if ch == '"':
                if in_quotes:
                    in_quotes = False
                else:
                    in_quotes = True
            if in_quotes:
                continue

            if ch == ":":
                idx = i
                while idx < len(penman_str) and penman_str[idx] != ' ':
                    idx += 1
                if idx+1 < len(penman_str) and penman_str[idx+1] != '(':
                    var = ''
                    j = idx+1
                    while j < len(penman_str) and penman_str[j] not in [' ','\n']:
                        var += penman_str[j]
                        j += 1
                    if var not in all_vars:
                        lnum = len(lvls)
                        if lnum >= len(cidx):
                            cidx.append(1)
                        else:
                            cidx[lnum] += 1                            
            if ch == '(':
                lnum = len(lvls)
                if lnum >= len(cidx):
                    cidx.append(0)
                lvls.append(str(cidx[lnum]))
            
            if ch == ')':
                lnum = len(lvls)
                if lnum < len(cidx):
                    cidx.pop()
                cidx[lnum-1] += 1
                lvls.pop()

            if ".".join(lvls) == node_id:
                j = i+1
                while penman_str[j] == ' ':
                    j += 1
                var = ""
                while penman_str[j] != ' ':
                    var += penman_str[j]
                    j += 1
                return var

        return None

    @classmethod
    def from_metadata(cls, penman_text, tokenize=False):
        """Read AMR from metadata (IBM style)"""

        # Read metadata from penman
        field_key = re.compile(f'::[A-Za-z]+')
        metadata = defaultdict(list)
        separator = None
        penman_str = ""
        for line in penman_text:
            if line.startswith('#'):
                line = line[2:].strip()
                start = 0
                for point in field_key.finditer(line):
                    end = point.start()
                    value = line[start:end]
                    if value:
                        metadata[separator].append(value)
                    separator = line[end:point.end()][2:]
                    start = point.end()
                value = line[start:]
                if value:
                    metadata[separator].append(value)
            else:
                penman_str += line.strip() + ' ' 
                    
        # assert 'tok' in metadata, "AMR must contain field ::tok"
        if tokenize:
            assert 'snt' in metadata, "AMR must contain field ::snt"
            tokens, _ = protected_tokenizer(metadata['snt'][0])
        else:
            assert 'tok' in metadata, "AMR must contain field ::tok"
            assert len(metadata['tok']) == 1
            tokens = metadata['tok'][0].split()

        #print(penman_str)
            
        sid="000"
        nodes = {}
        nvars = {}
        alignments = {}
        edges = []
        root = None
        sentence = None
    
        if 'short' in metadata:
            short_str = metadata["short"][0].split('\t')[1]
            short = eval(short_str)
            short = {str(k):v for k,v in short.items()}
            all_vars = list(short.values())
        else:
            short = None
            all_vars = AMR2.get_all_vars(penman_str)
        

        for key, value in metadata.items():
            if key == 'edge':
                for items in value:
                    items = items.split('\t')
                    if len(items) == 6:
                        _, _, label, _, src, tgt = items
                        edges.append((src, f':{label}', tgt))
            elif key == 'node':
                for items in value:
                    items = items.split('\t')
                    if len(items) > 3:
                        _, node_id, node_name, alignment = items
                        start, end = alignment_regex.match(alignment).groups()
                        indices = list(range(int(start), int(end)))
                        alignments[node_id] = indices
                    else:
                        _, node_id, node_name = items
                        alignments[node_id] = None
                    nodes[node_id] = node_name
                    if short is not None:
                        var = short[node_id]
                    else:
                        var = node_id
                    if var is not None and var+" / " not in penman_str:
                        nvars[node_id] = None
                    else:
                        nvars[node_id] = var
                        all_vars.remove(var)
            elif key == 'root':
                root = value[0].split('\t')[1]
            elif key == 'id':
                sid = value[0].strip()
        if len(all_vars):
            print("varaible not linked to nodes:")
            print(all_vars)
            print(penman_str)
        return cls(tokens, nodes, edges, root, penman=None,
                   alignments=alignments, nvars=nvars, sid=sid)

def read_amr2(file_path, ibm_format=False, tokenize=False):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = []
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                if ibm_format:
                    # From ::node, ::edge etc
                    raw_amrs.append(
                        AMR2.from_metadata(raw_amr, tokenize=tokenize)
                    )
                else:
                    # From penman
                    raw_amrs.append(
                        AMR.from_penman(raw_amr, tokenize=tokenize)
                    )
                raw_amr = []
            else:
                raw_amr.append(line)
    return raw_amrs

def read_amr3(file_path, ibm_format=False, tokenize=False):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = {}
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                if ibm_format:
                    # From ::node, ::edge etc
                    amr = AMR2.from_metadata(raw_amr, tokenize=tokenize)
                else:
                    # From penman
                    
                    amr = AMR.from_penman(raw_amr, tokenize=tokenize)

                raw_amrs[amr.sid] = amr
                raw_amr = []
            else:
                raw_amr.append(line)
    return raw_amrs

def read_amr3_docid(file_path, ibm_format=False, tokenize=False):
    doc_id = None
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = {}
        
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                if ibm_format:
                    # From ::node, ::edge etc
                    amr = AMR2.from_metadata(raw_amr, tokenize=tokenize)
                else:
                    # From penman
                    amr = AMR.from_penman(raw_amr, tokenize=tokenize)
                raw_amrs[amr.sid] = amr
                if doc_id is None:
                    doc_id = amr.sid.rsplit('.',1)[0]
                
                    
                raw_amr = []
            else:
                raw_amr.append(line)
    return raw_amrs,doc_id

#store by sen
def read_amr_by_snt(file_path, tokenize=False):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = {}
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                
                amr = AMR.from_penman(raw_amr, tokenize=tokenize)
                if tokenize:
                    tok_sen = " ".join(amr.tokens)
                    raw_amrs[tok_sen] = raw_amr
                else:
                    raw_amrs[amr.penman.metadata['tok']] = raw_amr
                raw_amr = []
            else:
                raw_amr.append(line)
    return raw_amrs

def read_amr_raw(file_path, tokenize=False):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = []
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                
                amr = AMR.from_penman(raw_amr, tokenize=tokenize)
                raw_amrs.append(raw_amr)
                raw_amr = []
            else:
                raw_amr.append(line)
    return raw_amrs

def read_amr_as_raw_str(file_path):
    with open(file_path) as fid:
        raw_amrs = []
        raw_amr = ''
        for line in fid.readlines():
            if line.strip():
                raw_amr+=line.strip()+'\n'
            else:
                raw_amrs.append(raw_amr)
                raw_amr = ''
    return raw_amrs

#for amrs without an ::id 
def read_amr_add_sen_id(file_path,doc_id,remove_id=False,tokenize=False,ibm_format=True):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = {}
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if line.strip() == '':
                if ibm_format:
                    # From ::node, ::edge etc
                    amr = AMR2.from_metadata(raw_amr, tokenize=tokenize)
                else:
                    # From penman
                    amr = AMR.from_penman(raw_amr, tokenize=tokenize)
                raw_amrs[amr.sid] = amr
                raw_amr = []
            else:
                if remove_id and '::id' in line:
                    continue
                raw_amr.append(line)
            if tokenize:
                if '::snt' in line and remove_id:
                    raw_amr.append('# ::id '+doc_id+'.'+str(len(raw_amrs)+1))
            else:
                if '::tok' in line and remove_id:
                    raw_amr.append('# ::id '+doc_id+'.'+str(len(raw_amrs)+1))
                elif '::snt' in line and remove_id:
                    raw_amr.append('# ::id '+doc_id+'.'+str(len(raw_amrs)+1))
            

    return raw_amrs

def read_amr_str_add_sen_id(amr_strs,doc_id,tokenize=False):
    raw_amrs = {}
    for idx,amr_str in enumerate(amr_strs):
    
        # From ::node, ::edge etc
        amr_list = amr_str.splitlines(True)
        amr_list.insert(1,'# ::id '+doc_id+'.'+str(idx+1)+'\n')
        # amr_list = [line+'\n# ::id '+doc_id+'.'+str(idx+1) if '::tok' in line else line for line in amr_str.split('\n')]
        amr = AMR2.from_metadata(amr_list, tokenize=tokenize)
        raw_amrs[amr.sid] = amr
            
    return raw_amrs

def recent_member_by_sent(chain,sid,doc_id):
    def get_sid_fromstring(string):
         
        sid = [int(s) for s in re.findall(r'\d+', string)]
        assert len(sid)==1
        return sid[0]

    sid = get_sid_fromstring(sid)    
    diff = lambda chain : abs(get_sid_fromstring(chain[0].split('.')[0]) - sid)
    ent = min(chain, key=diff)
    fix = False
    if get_sid_fromstring(ent[0].split('.')[0]) > sid:
    #     print(doc_id," closest sent is higher than connecting node ",ent[0],sid)
        fix = True
    return ent[0]

    

def recent_member_by_align(chain,src_align,doc_id,rel=None):
 
    diff = lambda chain : abs(chain[1]-src_align)
    ent = min(chain, key=diff)
    fix = False
    if ent[1]>= src_align:
    #     print(doc_id," coref edge missing ",ent[1],src_align,rel)
        fix  = True      
    return ent[0]

#convert v0 coref edge to connect to most recent sibling in the chain
def make_pairwise_edges(damr,verbose=False):
    
    ents_chain = defaultdict(list)
    edges_to_delete = []
    nodes_to_delete = []
    doc_id = damr.amr_id
    # damr.edges.sort(key = lambda x: x[0])
    for idx,e in enumerate(damr.edges):
        if e[1] == ':coref-of':
            # if len(ents[e[2]])==0:
                #damr.edges[idx] = (e[0],':same-as',ents[e[2]][-1])
            # else:
            edges_to_delete.append(e)

            if e[0] in damr.alignments and damr.alignments[e[0]] is not None:
                ents_chain[e[2]].append((e[0],damr.alignments[e[0]][0]))
            else:
                #FIXME adding the src node of a coref edge with no alignments member of chain with closest sid
                # print(doc_id + '  ',e[0],' alignments is None  src node in coref edge, not adding it ')
                sid = e[0].split('.')[0]
                if len(ents_chain[e[2]]) >0 :
                    ent = recent_member_by_sent(ents_chain[e[2]],sid,doc_id)
                    damr.edges[idx] = (e[0],':same-as',ent)
                #FIXME
                else:
                    
                    print("coref edge missing, empty chain, edge not added")
                
            assert e[2].startswith('rel')
       

    
    #adding coref edges between most recent sibling in chain    
    for cents in ents_chain.values():
        cents.sort(key=lambda x:x[1])
        for idx in range(0,len(cents)-1):
            damr.edges.append((cents[idx+1][0],':same-as',cents[idx][0]))

    for e in edges_to_delete:
        while e in damr.edges:
            damr.edges.remove(e)

    #connecting all other edges involving chain to most recent member in the chain
    for idx,e in enumerate(damr.edges):
        #Both src and target are coref nodes
        if e[0] in ents_chain and e[2] in ents_chain:
            damr.edges[idx] = (ents_chain[e[0]][-1][0],e[1],ents_chain[e[2]][-1][0])
        
        elif e[2] in ents_chain.keys():
            #src node is a normal amr node
            if e[0] in damr.alignments and damr.alignments[e[0]] is not None:
                ent = recent_member_by_align(ents_chain[e[2]],damr.alignments[e[0]][0],doc_id,e[1])
                
            else:
                #FIXME assigning src node with no alignments to the recent member by sent in the coref chain
                # print(doc_id + '  ',e[0],' alignments is None ')
                sid = e[0].split('.')[0]
                ent = recent_member_by_sent(ents_chain[e[2]],sid,doc_id)
            damr.edges[idx] = (e[0],e[1],ent)

        elif e[0] in ents_chain.keys():
            if e[2] in damr.alignments and damr.alignments[e[2]] is not None:
                ent = recent_member_by_align(ents_chain[e[0]],damr.alignments[e[2]][0],doc_id,e[1])
            else:
                #FIXME assigning tgt node with no alignments to the recent member by sent in the coref chain
                # print(doc_id + '  ',e[0],' alignments is None ')
                sid = e[2].split('.')[0]
                ent = recent_member_by_sent(ents_chain[e[0]],sid,doc_id)
        
            damr.edges[idx] = (ent,e[1],e[2])

       
    for n in ents_chain.keys():
        while n in damr.nodes:
            del damr.nodes[n]
    
        
    
    return damr