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
                        var = AMR2.get_node_var(penman_str, node_id)
                    if var is not None and var+" / " not in penman_str:
                        nvars[node_id] = None
                    else:
                        nvars[node_id] = var
                    if var != None:
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
