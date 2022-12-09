import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
import penman
from penman.layout import Push
from penman import surface
import copy
from collections import Counter, defaultdict

def normalize_tok(token):
    """
    Normalize token or node
    """
    if token == '"':
        return token
    else:
        return token.replace('"', '')

def simple_tokenizer(string):
    ret_str = re.sub(r'([a-zA-Z])(\'[a-zA-Z])',r'\1 \2',string)
    ret_str = re.sub(r'([\.,;:?!"\(\)\[\]\{\}])',r' \1 ',ret_str)
    return ret_str.split()

def read_amr(file_path, tokenize=False):
    with open(file_path) as fid:
        raw_amr = []
        raw_amrs = {}
        for line in tqdm(fid.readlines(), desc='Reading AMR'):
            if "# AMR release" in line:
                continue
            if line.strip() == '':
                if len(raw_amr):
                    # From penman
                    amr = AMR.from_penman(raw_amr, tokenize=tokenize)
                    raw_amrs[amr.sid] = amr
                    raw_amr = []                
            else:
                raw_amr.append(line)
                
        if len(raw_amr):
            # From penman
            amr = AMR.from_penman(raw_amr, tokenize=tokenize)
            raw_amrs[amr.sid] = amr
            raw_amr = []

    return raw_amrs

def get_simple_graph(graph):
    """
    Get simple nodes/edges/alignments representation from penman class
    """

    # alignments
    isi_alignments = surface.alignments(graph)

    # get map of node variables to node names (this excludes constants)
    name_to_node = {}
    alignments = {}
    for x in graph.instances():
        name_to_node[x.source] = x.target
        if x in isi_alignments:
            if len(isi_alignments[x].indices) == 1:
                alignments[x.source] = list(isi_alignments[x].indices)
            elif len(isi_alignments[x].indices) == 2:
                start = isi_alignments[x].indices[0]
                end = isi_alignments[x].indices[-1]
                alignments[x.source] = list(range(start, end))
            else:
                raise Exception('Unexpected ISI alignment format')

    # reentrancy
    reentrancies = [e for e, c in graph.reentrancies().items() if c > 1]

    # Get all edges (excludes constants)
    edges = []
    re_entrant = defaultdict(list)
    for x in graph.edges():
        assert x.target in name_to_node

        # get epidata
        edge_epidata = graph.epidata[(x.source, x.role, x.target)]

        # keep inverted edges
        if (
            edge_epidata
            and isinstance(edge_epidata[0], Push)
            and edge_epidata[0].variable == x.source
        ):
            # reversed edge
            edge = (x.target, f'{x.role}-of', x.source)
        else:
            edge = (x.source, x.role, x.target)

        # delay adding an edge if its reentrant until we produce the original
        # edge
        # TODO: Does not work fully, remove?
        if (
            edge[-1] in reentrancies
            and edge_epidata
            and isinstance(edge_epidata[0], Push)
            and edge_epidata[0].variable == edge[-1]
            and re_entrant[edge[-1]] is not None
        ):
            # this is the original edge from a re-entrant series
            # append edge and all the re-entrancies
            edges.append(edge)
            edges.extend(re_entrant[edge[-1]])
            # block, since we do not need it any more
            re_entrant[edge[-1]] = None

        elif (
            edge[-1] in reentrancies
            and re_entrant[edge[-1]] is not None
        ):
            # append also edges rentrant to these
            re_entrant[edge[-1]].append(edge)

        else:
            # append edge
            edges.append(edge)

    # if nodes are re-entran to root, we will reach here with pending edges
    for nid, rest in re_entrant.items():
        if rest is not None:
            edges.extend(rest)

    if len(edges) < len(graph.edges()):
        set_trace(context=30)

    # Add constants both to node map and edges, use position in attribute as id
    # sort attributes by target for consistency in assignment
    attributes = sorted(
        graph.attributes(),
        key=lambda x: (
            x.source, x.role, x.target.replace('"', '')
        )
    )
    index = 0
    attribute_nodes = []
    for att in attributes:
        assert index not in name_to_node
        # will be used as a node id, needs to be a string
        # watch for existing numbers used as ids
        while str(index) in name_to_node:
            index += 1
        att_id = str(index)
        name_to_node[att_id] = att.target
        attribute_nodes.append(att_id)
        # add alignments
        if att in isi_alignments:
            if len(isi_alignments[att].indices) == 1:
                alignments[att_id] = list(isi_alignments[att].indices)
            elif len(isi_alignments[att].indices) == 2:
                start = isi_alignments[att].indices[0]
                end = isi_alignments[att].indices[-1]
                alignments[att_id] = list(range(start, end))
            else:
                raise Exception('Unexpected ISI alignment format')

        edge_epidata = graph.epidata[(att.source, att.role, att.target)]
        if (
            edge_epidata
            and isinstance(edge_epidata[0], Push)
            and edge_epidata[0].variable == x.source
        ):
            # reversed edge
            raise Exception()
            edges.append((att_id, f'{att.role}-of', att.source))
        else:
            edges.append((att.source, att.role, att_id))

        # increase index
        index += 1

    return name_to_node, edges, alignments, attribute_nodes

def process_corefs(fnames):

    all_corefs = {}
    
    for fname in tqdm(fnames, desc="Reading Coref XML files"):
        tree = ET.parse(fname)
        root = tree.getroot()
        doc_id = root[0].attrib['docid']
        doc_sen_ids = []
        order = {}
        for sen in root[0]:
            doc_sen_ids.append(sen.attrib['id'])
            order[sen.attrib['id']] = sen.attrib['order']
            
        document_chains = {}
        document_bridges = {}
        document_sngltns = {}

        #identity relations
        for i in range(len(root[1][0])):
            chain = []
            chain_id = root[1][0][i].attrib['relationid']
            for m in range(len(root[1][0][i])):
                sen_id = root[1][0][i][m].attrib['id']
                var = None
                if 'variable' in root[1][0][i][m].attrib:
                    var = root[1][0][i][m].attrib['variable']
                else:
                    var = root[1][0][i][m].attrib['parentvariable']
                rel = AMR.coref_rel_inv
                if root[1][0][i][m].tag == 'implicitrole':
                    rel = ":"+root[1][0][i][m].attrib['argument']
                chain.append((sen_id+"."+var,rel))
            document_chains[chain_id] = chain

        #singltons
        for i in range(len(root[1][1])):
            sen_id = root[1][1][i][0].attrib['id']
            singleton_id = root[1][1][i].attrib['relationid']
            rel = "SELF"
            if 'variable' not in root[1][1][i][0].attrib.keys():
                var = root[1][1][i][0].attrib['parentvariable']
                rel = ":"+root[1][1][i][0].attrib['argument']
            else:
                var = root[1][1][i][0].attrib['variable']
            document_sngltns[singleton_id] = (sen_id +"." + var, rel)

        #bridges
        for i in range(len(root[1][2])):
            parent = ""
            parent_rel = ""
            kid_rel = ""
            kids = []
            bridge_id = root[1][2][i].attrib['relationid']
            for m in range(len(root[1][2][i])):
                rel_id = root[1][2][i][m].attrib['id']
                if rel_id not in document_chains and rel_id not in document_sngltns:
                    print('missing '+rel_id+' in '+fname)
                    continue
                tag = root[1][2][i][m].tag
                if tag in ['superset','whole']:
                    parent = rel_id
                    parent_rel = ":"+root[1][2][i][m].tag
                else:
                    kids.append(rel_id)
                    kid_rel = ":"+root[1][2][i][m].tag
            for kid in kids:
                if len(parent) == 0 :
                    print("missing parent")
                    break
                document_bridges[bridge_id] = (kid, kid_rel, parent)

        all_corefs[doc_id] = ((document_chains, document_sngltns, document_bridges), doc_sen_ids,fname)

    return all_corefs


alignment_regex = re.compile('(-?[0-9]+)-(-?[0-9]+)')
#for document level AMR anaphora merging
pronouns = ['i','we','you','he','she','that','this','these','those','it','they','all','another','anybody','anyone','anything','any','both','each','either','everybody','everyone','everything','neither','nobody','none','nothing','one','few','most','many','other','several','some','somebody','someone','something','such','whatever','whichever','whoever','whomever']
#personal_pronouns = ['i', 'we', 'you', 'he', 'she', 'they', 'one', 'someone', 'self']

#NE types ordered from specific to general
NETypes = ['medical-condition', 'disease', 'taxon', 'species', 'cell-line', 'cell', 'dna-sequence', 'gene', 'pathway', 'nucleic-acid', 'enzyme', 'macro-molecular-complex', 'amino-acid', 'protein-segment', 'protein-family', 'protein', 'small-molecule', 'molecular-physical-entity', 'program', 'variable', 'writing-script', 'food-dish', 'musical-note', 'music-key', 'treaty', 'court-decision', 'law', 'award', 'natural-object', 'journal', 'magazine', 'newspaper', 'book', 'publication', 'broadcast-program', 'show', 'music', 'picture', 'work-of-art', 'car-make', 'spaceship', 'aircraft-type', 'aircraft', 'ship', 'vehicle', 'product', 'festival', 'game', 'conference', 'war', 'earthquake', 'natural-disaster', 'incident', 'event', 'amusement-park', 'zoo', 'park', 'market', 'sports-facility', 'worship-place', 'hotel', 'palace', 'museum', 'theater', 'building', 'canal', 'railway-line', 'road', 'bridge', 'tunnel', 'port', 'station', 'airport', 'facility', 'constellation', 'star', 'planet', 'moon', 'forest', 'desert', 'island', 'canyon', 'valley', 'volcano', 'mountain', 'peninsula', 'canal', 'strait', 'bay', 'gulf', 'river', 'lake', 'sea', 'ocean', 'continent', 'world-region', 'country-region', 'local-region', 'country', 'territory', 'province', 'state', 'county', 'city-district', 'city', 'location', 'league', 'team', 'research-institute', 'university', 'school', 'market-sector', 'political-party', 'criminal-organization', 'military', 'government-organization', 'company', 'organization', 'political-movement', 'religious-group', 'regional-group', 'ethnic-group', 'nationality', 'language', 'animal', 'family', 'person', 'thing']

class AMR():

    id_counter = 0

    coref_node = "coref-entity"
    coref_rel = ":coref"
    coref_rel_inv = coref_rel+"-of"
    pro_node = "pro"
    verbose = True
    
    def __init__(self, tokens, nodes, edges, root, penman=None,
                 alignments=None, sid = None, nvars = None, sentence_ends=None):

        if sid is None:
            sid = str(AMR.id_counter)
            AMR.id_counter += 1
        self.sid = sid
        self.nvars = nvars
        
        # make graph un editable
        self.tokens = tokens
        self.nodes = nodes
        self.edges = edges
        self.penman = penman
        self.alignments = alignments

        self.sentence_ends = sentence_ends

        # root
        self.root = root
        self.roots = []
        if self.nodes[self.root] == 'document':
            for (s,rel,t) in self.edges:
                if s == self.root and rel.startswith(':snt'):
                    self.roots.append(t)
                    
        # xml file name, for doc-amr
        self.doc_file = None
        self.amr_id = sid
        
        if self.nvars:
            self.nvars2nodes = {}
            for nid in self.nvars:
                if self.nvars[nid] != None:
                    self.nvars2nodes[self.nvars[nid]] = nid
        self._cache_key = None

    def __add__(self, other):
        
        if len(self.roots) == 0:

            new_nodes = {}
            new_nvars = {}
            new_edges = []
            new_alignments = {}
            
            for nid in self.nodes:
                new_nid = "s1."+str(nid)
                new_nodes[new_nid] = self.nodes[nid]
                if self.sid and self.nvars:
                    new_nvars[new_nid] = self.sid+"."+self.nvars[nid] if self.nvars[nid] is not None else None
                if self.alignments:
                    if nid in self.alignments:
                        new_alignments[new_nid] = self.alignments[nid]
            for (n1,l,n2) in self.edges:
                new_edges.append(("s1."+str(n1), l, "s1."+str(n2)))
            
            self.nodes = new_nodes
            self.edges = new_edges
            if self.alignments:
                self.alignments = new_alignments
            if self.nvars:
                self.nvars = new_nvars
            
            self.roots = ['s1.'+self.root]
            self.root = None

            self.sidx2nodes = {}
            self.nodes2sidx = {}

            self.sidx2nodes[0] = []
            for nid in self.nodes:
                self.sidx2nodes[0].append(nid)
                self.nodes2sidx[nid] = 0
                    
            

        self.tokens.append('<next_sent>')
        start_idx = len(self.tokens)
        sen_idx   = len(self.roots)+1
        self.tokens += other.tokens

        self.sidx2nodes[sen_idx] = []
        
        for nid in other.nodes:
            new_nid = "s"+str(sen_idx)+"."+str(nid)
            self.nodes[new_nid] = other.nodes[nid]
            if self.sid and self.nvars:
                self.nvars[new_nid] = other.sid+"."+other.nvars[nid] if other.nvars[nid] is not None else None
            if self.alignments:
                if nid in other.alignments and other.alignments[nid] is not None:
                    self.alignments[new_nid] = [aln + start_idx for aln in other.alignments[nid]]
                else:
                    self.alignments[new_nid] = None
                    
            self.sidx2nodes[sen_idx].append(new_nid)
            self.nodes2sidx[new_nid] = sen_idx
        for (n1,l,n2) in other.edges:
            self.edges.append(("s"+str(sen_idx)+"."+str(n1), l, "s"+str(sen_idx)+"."+str(n2)))
        
        if self.sid and self.nvars:
            self.sid = self.sid+"_"+other.sid
            self.nvars2nodes = {}
            for nid in self.nvars:
                if self.nvars[nid] != None:
                    self.nvars2nodes[self.nvars[nid]] = nid
        
        self.roots.append("s"+str(sen_idx)+"."+other.root)
        
        return self

    def cache_graph(self):
        '''
        Precompute edges indexed by parent or child
        '''

        # If the cache has not changed, no need to recompute
        if self._cache_key == tuple(self.edges):
            return

        # edges by parent
        self._edges_by_parent = defaultdict(list)
        for (source, edge_name, target) in self.edges:
            self._edges_by_parent[source].append((target, edge_name))

        # edges by child
        self._edges_by_child = defaultdict(list)
        for (source, edge_name, target) in self.edges:
            self._edges_by_child[target].append((source, edge_name))

        # store a key to know when to recompute
        self._cache_key == tuple(self.edges)

    def parents(self, node_id, edges=True):
        self.cache_graph()
        arcs = self._edges_by_child.get(node_id, [])
        if edges:
            return arcs
        else:
            return [a[0] for a in arcs]

    def children(self, node_id, edges=True):
        self.cache_graph()
        arcs = self._edges_by_parent.get(node_id, [])
        if edges:
            return arcs
        else:
            return [a[0] for a in arcs]

    def add_node(self, form, prefix=None, is_constant=False):

        num = 0
        if prefix is None:
            prefix = form[0]

        node_id = prefix
        while node_id in self.nodes:
            num += 1
            node_id = prefix+str(num)

        self.nodes[node_id] = form

        if is_constant:
            self.nvars[node_id] = None
        else:
            num = 0
            var = prefix
            while var in self.nvars2nodes:
                num += 1
                var = prefix+str(num)
            self.nvars[node_id] = var
            self.nvars2nodes[var] = node_id

        return node_id    

    def merge_nodes(self, node1, node2, additional=False, merge_kids=True):

        if node1 not in self.nodes or node2 not in self.nodes:
            return
        '''
        if self.nodes[node1] != self.nodes[node2] and self.nodes[node2] != AMR.coref_node:
            print("merging:")
            print(self.get_sub_str(node1))
            print(self.get_sub_str(node2))
            print("---------")
        '''
        node1_forms = [self.nodes[node1]]
        node1_names = []
        node1_wikis = []
        node1_kids = {}
        for (i,e) in enumerate(self.edges):
            if e[0] == node1:
                if e[1] == ":name":
                    node_name = self.get_name_str(e[2])
                    if node_name not in node1_names:
                        node1_names.append(node_name)
                elif e[1] == ":wiki":
                    node1_wikis.append(self.nodes[e[2]])
                elif e[1] == ":additional-type" and self.nodes[e[2]] not in node1_forms:
                    node1_forms.append(self.nodes[e[2]])
                else:
                    node1_kids[(e[1],self.nodes[e[2]])] = e[2]
                    
        edges_to_delete = []
        mergeable_kids = []
        for (i,e) in enumerate(self.edges):
            if e in edges_to_delete:
                continue
            if e[2] == node2:
                if e[0] == node1 or e[0] == node2:
                    edges_to_delete.append(e)
                    continue
                new_edge = (e[0], e[1], node1)
                if new_edge not in self.edges:
                    self.edges[i] = new_edge
                else:
                    edges_to_delete.append(e)
            if e[0] == node2:
                if e[2] == node1 or e[2] == node2:
                    edges_to_delete.append(e)
                    continue
                if e[1] not in [":name",":wiki"]:
                    #see if the kid should be merged recursively
                    if e[2] not in self.nodes:
                        edges_to_delete.append(e)
                        print("edge node does not exist !!!") 
                    else:
                        kid_form = (e[1],self.nodes[e[2]])
                        if kid_form in node1_kids and node1_kids[kid_form] != e[2]:
                            pair = (node1_kids[(e[1],self.nodes[e[2]])], e[2])
                            if pair not in mergeable_kids:
                                mergeable_kids.append(pair)
                        #assign the kid to node1
                        new_edge = (node1, e[1], e[2])
                        if new_edge not in self.edges:
                            self.edges[i] = new_edge
                        else:
                            edges_to_delete.append(e)
                        
                elif e[1] == ":name":
                    new_node_name = self.get_name_str(e[2])
                    if new_node_name not in node1_names:
                        self.edges[i] = (node1, ":name", e[2])
                    else:
                        has_other_parents = False
                        for (x,r,y) in self.edges:
                            if y == e[2]:
                                if x != e[0]:
                                    has_other_parents = True
                                elif r != e[1]:
                                    edges_to_delete.append((x,r,y))
                        if not has_other_parents:
                            left_overs = self.delete_name(e[2])
                            for (r,n) in left_overs:
                                self.edges.append((node1, r, n))                            
                        edges_to_delete.append(e)
                else:
                    new_node_wiki = self.nodes[e[2]]
                    if new_node_wiki not in node1_wikis and new_node_wiki != "-":
                        self.edges[i] = (node1, e[1], e[2])
                        all_wikis = new_node_wiki
                        for wiki in node1_wikis:
                            if wiki != '-':
                                all_wikis += " " + wiki
                        if ' ' in all_wikis and AMR.verbose:
                            print("Merged nodes have different wikis")
                            print(all_wikis)
                    else:
                        has_other_parents = False
                        for (x,r,y) in self.edges:
                            if y == e[2] and x != e[0]:
                                has_other_parents = True
                        if not has_other_parents:
                            del self.nodes[e[2]]
                        edges_to_delete.append(e)
                                        
        if additional and self.nodes[node2] not in node1_forms:
            self.edges.append((node1,":additional-type",node2))
            #self.nvars[node2] = None #ensure that this node will be treated as attribute
        else:
            if node2 not in self.nodes:
                print("node2 disappeared !!!")
            del self.nodes[node2]

        for e in edges_to_delete:
            while e in self.edges:
                self.edges.remove(e)
                
        if merge_kids:
            for (n1,n2) in mergeable_kids:
                self.merge_nodes(n1,n2,additional)


    def add_corefs(self, corefs, annotated=False):

        (chains, singletons, bridges) = corefs
        
        chains2nodes = {}
        for ch_id in chains:
            chain = chains[ch_id]
            chain_nid = self.add_node(AMR.coref_node, prefix=ch_id)
            chains2nodes[ch_id] = chain_nid
            for (nvar,rel) in chain:
                node = self.nvars2nodes[nvar]
                if annotated and rel != AMR.coref_rel_inv:
                    rel = rel + "-implicit"
                edge = (node,rel,chain_nid)
                if edge not in self.edges:
                    self.edges.append(edge)
                    
        singletons2nodes = {}
        for sg_id in singletons:
            singleton = singletons[sg_id]
            rel = singleton[1]
            node = self.nvars2nodes[singleton[0]]
            if rel != 'SELF':
                pnode = node
                node = self.add_node("implicit-role", prefix=node+"i")
                if not annotated:
                    self.edges.append( (pnode, rel, node) )
                else:
                    self.edges.append( (pnode, rel+"-implicit", node) )
            singletons2nodes[sg_id] = node

        for bid in bridges:
            bridge = bridges[bid]
            if bridge[0] in chains2nodes:
                node1 = chains2nodes[bridge[0]]
            if bridge[2] in chains2nodes:
                node2 = chains2nodes[bridge[2]]
            if bridge[0] in singletons2nodes:
                node1 = singletons2nodes[bridge[0]]
            if bridge[2] in singletons2nodes:
                node2 = singletons2nodes[bridge[2]]
            rel = bridge[1]
            if rel == ":part":
                edge = (node1,":part-of",node2)
            if rel == ":member":
                edge = (node1,":subset-of",node2)
            if edge not in self.edges:
                self.edges.append(edge)

    def un_invert(self):
        for (i,e) in enumerate(self.edges):
            if e[1].endswith("-of") and e[1] != AMR.coref_rel_inv:
                self.edges[i] = (e[2],e[1][:-3],e[0])
                
    def normalize(self, rep='docAMR', flip=False):

        #self.un_invert()
        
        if rep == 'no-merge':
            self.remove_one_node_chains()
            return
        
        if rep == 'merge-names':
            self.merge_names(additional=True)
            self.remove_one_node_chains()
            
        if rep == 'docAMR':
            self.merge_names(additional=True)
            self.merge_anaphora()
            self.remove_one_node_chains()
            
        if flip:
            self.upside_down()
                        
        if rep == 'merge-all':
            self.merge_all(additional=True)
            self.remove_one_node_chains()

    def get_chain_nodes(self):
        chain2nodes = {}
        for e in self.edges:
            if e[1] == AMR.coref_rel_inv:
                if e[2] not in chain2nodes:
                    chain2nodes[e[2]] = []
                chain2nodes[e[2]].append(e[0])
            if e[1] == AMR.coref_rel:
                if e[0] not in chain2nodes:
                    chain2nodes[e[0]] = []
                chain2nodes[e[0]].append(e[2])
        return chain2nodes
    
    def remove_one_node_chains(self):

        chain2nodes = self.get_chain_nodes()

        for chain_id in chain2nodes:
            if len(chain2nodes[chain_id]) == 1:
                #only one node left in chains, no need for chain
                only_node = chain2nodes[chain_id][0]
                self.merge_nodes(only_node, chain_id)
                
    def merge_all(self, additional=False):

        chain2nodes = self.get_chain_nodes()

        #merging all nodes in chain
        for chain_id in chain2nodes:
            chain_head = chain2nodes[chain_id][0]
            for nid in chain2nodes[chain_id][1:]:
                self.merge_nodes(chain_head, nid, additional)            

    def merge_names(self, additional=False):

        chain2nodes = self.get_chain_nodes()
        
        #merging all nodes in chain
        for chain_id in chain2nodes:
            named_nodes = []
            name_types = []
            for nid in chain2nodes[chain_id]:
                for e in self.edges:
                    if e[0] == nid and e[1] == ":name":
                        named_nodes.append(nid)
                        name_types.append(self.nodes[nid])
                        #for ee in self.edges:
                        #    if ee[0] == nid and ee[1] not in [":name",":wiki",":subset-of",":member-of"] and ee[2] != AMR.coref_node:
                        #        print(chain_id+" other kid: "+self.nodes[ee[0]]+" "+ee[1]+" "+self.get_sub_str(ee[2]))
                        break
            if len(named_nodes):
                names_head = named_nodes[0]
                if len(set(name_types)) > 1:
                    if AMR.verbose:
                        print("multiple types: "+", ".join(name_types))
                    found_amr_type = False
                    for ntype in NETypes:
                        if ntype in name_types:
                            idx = name_types.index(ntype)
                            names_head = named_nodes[idx]
                            found_amr_type = True
                            if AMR.verbose:
                                print("type selected based on NE types " + ntype)
                            break
                    if not found_amr_type:
                        freq_type = sorted(Counter(name_types).items(), key=lambda tup: tup[1], reverse=True)[0]
                        idx = name_types.index(freq_type)
                        names_head = named_nodes[idx]
                        if AMR.verbose:
                            print("type selected based on frequency " + freq_type)
                        
                for nid in named_nodes:
                    if nid != names_head:
                        self.merge_nodes(names_head, nid, additional)            
                self.remove_extra_empty_wikis(names_head)

    def get_sub_str(self, node):
        kids = self.children(node)
        kid_forms = " ".join(sorted([self.nodes[x[0]]+x[1] for x in kids]))
        for (i,kid) in enumerate(kids):
            kid_forms += " | " + get_sub_str(kid[0])

    def remove_extra_empty_wikis(self, node):

        empty_wiki = None
        other_wikis = []
        for e in self.edges:
            if e[0] == node and e[1] == ":wiki":
                if self.nodes[e[2]] == "-":
                    empty_wiki = e[2]
                else:
                    other_wikis.append(e[2])

        if empty_wiki is not None:
            if len(other_wikis):
                self.edges.remove( (node,":wiki",empty_wiki) )

                
    def merge_sub(self, node1, node2):
        kids1 = self.children(node1)
        kids2 = self.children(node2)
        kid_forms1 = [self.get_sub_str(x[0]) for x in kids1]
        kid_forms2 = [self.get_sub_str(x[0]) for x in kids2]
        for (i,kid) in enumerate(kids1):
            if kid_forms1[i] in kid_forms2:
                j = kid_forms2.index(kid_forms1[i])
                if kid[0] != kids2[j][0]:
                    if kids2[j][0] not in self.nodes:
                        print("cant merge, node does not exist")
                    else:
                        self.merge_sub(kid[0],kids2[j][0])
                
        self.merge_nodes(node1, node2)
            
    def merge_same(self):

        chain2nodes = self.get_chain_nodes()

        #merging same forms
        for chain_id in chain2nodes:
            all_forms = set()
            for nid in chain2nodes[chain_id]:
                all_forms.add(self.nodes[nid])
            if len(all_forms) == 1:
                chain_head = chain2nodes[chain_id][0]
                for nid in chain2nodes[chain_id][1:]:
                    self.merge_sub(chain_head, nid)
                
            
                
    def merge_anaphora(self):

        chain2nodes = self.get_chain_nodes()

        #merging anaphora
        for chain_id in chain2nodes:
            pron_forms = set()
            othr_forms = set()
            chain_prons = []
            chain_othrs = []
            for nid in chain2nodes[chain_id]:
                if nid not in self.nodes:
                    continue
                if self.nodes[nid] in pronouns:
                    chain_prons.append(nid)
                    pron_forms.add(self.nodes[nid])
                else:
                    chain_othrs.append(nid)
                    othr_forms.add(self.nodes[nid])

            if len(chain_prons) == 0:
                continue
            
            #merge all pronouns into non-pronoun node
            chain_head = None
            if len(chain_othrs):
                chain_head = chain_id
            elif len(pron_forms) > 1:
                if "i" in pron_forms or "you" in pron_forms:
                    chain_head = self.add_node('interlocutor-entity',prefix='pro')
                else:
                    for pro in pronouns:
                        if pro in pron_forms:
                            chain_head = self.add_node(pro,prefix='pro')
                            break
            else:
                chain_head = self.add_node(list(pron_forms)[0],prefix='pro')

            if AMR.verbose:
                print("pronouns in the chain: " + " ".join(pron_forms))
                print("other nodes in the chain: " + " ".join(othr_forms))
                if len(othr_forms) != 1:
                    print("Chain Head (" + chain_head + "): " + self.nodes[chain_head])
                else:
                    #this case is taken care of later when singlen node  chains are dropped
                    print("Chain Head (" + str(chain_othrs[0]) + "): " + " ".join(othr_forms))
                print("--------------------------")
            
            for pid in chain_prons:
                if pid != chain_head:
                    self.merge_nodes(chain_head, pid, merge_kids=False)

    def merge_anaphora_old(self):

        chain2nodes = self.get_chain_nodes()

        #merging anaphora
        for chain_id in chain2nodes:
            pron_forms = set()
            othr_forms = set()
            chain_prons = []
            chain_othrs = []
            for nid in chain2nodes[chain_id]:
                if nid not in self.nodes:
                    continue
                if self.nodes[nid] in personal_pronouns:
                    chain_prons.append(nid)
                    pron_forms.add(self.nodes[nid])
                else:
                    chain_othrs.append(nid)
                    othr_forms.add(self.nodes[nid])

            if len(chain_prons) == 0:
                continue
            
            #merge all pronouns into non-pronoun node
            name = ""
            found_head = False
            if len(chain_othrs):
                if len(othr_forms) == 1:
                    chain_head = chain_othrs[0]
                    found_head = True
                else:
                    for nid in chain_othrs:
                        for e in self.edges:
                            if e[0] == nid and e[1] == ":name":
                                name = "("+self.get_name_str(e[2])+")"
                                chain_head = nid
                                found_head = True
                                break

            if not found_head and len(chain_prons) > 1:
                found_head = True
                if len(pron_forms) > 1:
                    chain_head = self.add_node('interlocutor-entity',prefix='pro')
                else:
                    chain_head = chain_prons[0]

            out_str = ""
            for nid in chain2nodes[chain_id]:
                if nid in self.nodes:
                    out_str += self.nodes[nid]+" "

            if not found_head:
                if len(chain2nodes[chain_id]) > 1:
                    print("cannot find a good head " + chain_id + " {"+out_str+"}")
                else:
                    print("single node chain")
            else:                                  
                print("merging " + chain_id + " {"+out_str+"} into "+self.nodes[chain_head]+" "+name)
                for pid in chain_prons:
                    if pid != chain_head:
                        self.merge_nodes(chain_head, pid)

                if len(othr_forms) == 1:
                    for nid in chain_othrs[1:]:
                        if nid != chain_head:
                            if self.get_sub_str(chain_head) != self.get_sub_str(nid):
                                print("hard to merge same form nominals {" + out_str + "}")
                                print(self.get_sub_str(chain_head))
                                print(self.get_sub_str(nid))
                                print("")
                            self.merge_nodes(chain_head, nid)                    
                            
    def upside_down(self):

        chain2nodes = self.get_chain_nodes()

        #turning chains upside doen
        for chain_id in chain2nodes:
            for nid in chain2nodes[chain_id]:
                for (i,e) in enumerate(self.edges):
                    if e[2] == nid and not e[1].endswith("-of"):
                        self.edges[i] = (e[0],e[1],chain_id)
                    if e[0] == nid and e[1].endswith("-of") and e[1] != AMR.coref_rel_inv:
                        self.edges[i] = (chain_id,e[1],e[2])
        
    def get_name_str(self, name_node_id):
        if self.nodes[name_node_id] != 'name':
            return ''
        name_str = ''
        name_ops = {}
        max_op = -1
        for e in self.edges:
            if e[0] == name_node_id and e[1].startswith(':op'):
                name_ops[int(e[1][3:])] = self.nodes[e[2]].strip("\"")
                if int(e[1][3:]) > max_op:
                    max_op = int(e[1][3:])
        for i in range(1,max_op+1):
            if i in name_ops:
                name_str += name_ops[i]+" "

        return name_str.strip()

    def delete_name(self, name_node_id):
        if name_node_id not in self.nodes:
            print("can not delete, does not exist")
            return
        if self.nodes[name_node_id] != 'name':
            print("not a name node, deleteing anyways")
        edges_to_delete = []
        left_overs = []
        for e in self.edges:
            if e[0] == name_node_id:
                if 'op' in e[1]:
                    if e[2] in self.nodes:
                        del self.nodes[e[2]]
                else:
                    left_overs.append((e[1],e[2]))
                edges_to_delete.append(e)
        del self.nodes[name_node_id]
        for e in edges_to_delete:
            if e in self.edges:
                self.edges.remove(e)

        return left_overs

    def delete_sub(self, name_node_id):
        if name_node_id not in self.nodes:
            print("can not delete, does not exist")
            return
        for e in self.edges:
            if e[0] == name_node_id:
                if e[2] in self.nodes:
                    self.delete_sub(e[2])
        del self.nodes[name_node_id]

    def get_sub_str(self, node_id, local=False):
        if node_id not in self.nodes:
            print("can not print, does not exist")
            return
        ret_str = self.nodes[node_id] + " "
        for e in self.edges:
            if e[0] == node_id:
                if e[2] in self.nodes and not e[1].endswith("-of"):
                    ret_str += e[1] + " "
                    if local:
                        ret_str += "( " + self.nodes[e[2]] + " )"
                    else:
                        ret_str += "( " + self.get_sub_str(e[2], local=True) + " )"

        return ret_str
        
    @classmethod
    def from_penman(cls, penman_text, tokenize=False):
        """
        Read AMR from penman notation (will ignore graph data in metadata)
        """

        #assert isinstance(penman_text, str), "Expected string with EOL"
        #assert '\n' in penman_text, "Expected string with EOL"
        
        graph = penman.decode(penman_text)
        nodes, edges, alignments, attributes = get_simple_graph(graph)
        tokens = ""
        sid = None
        if 'id' in graph.metadata:
            sid=graph.metadata['id']
            
        if 'tok' in graph.metadata:
            tokens = graph.metadata['tok'].split()
        elif 'snt' in graph.metadata:
            if tokenize:
                tokens = simple_tokenizer(graph.metadata['snt'])
            else:
                tokens = [graph.metadata['snt']]

        # wipe out JAMR notation from metadata since it can become inconsistent
        # also remove unsupported "alignments" field
        delete_keys = []
        for key, data in graph.metadata.items():
            for okey in ['node', 'edge', 'root', 'short', 'alignments']:
                if key.startswith(okey):
                    delete_keys.append(key)
        for key in delete_keys:
            del graph.metadata[key]

        # remove quotes
        nodes = {nid: normalize_tok(nname) for nid, nname in nodes.items()}
                
        #node variables
        #values and literals dont get node variables
        nvars = dict.fromkeys(nodes, None)
        for nvar in nvars:
            if isinstance(nvar, str) and nvar not in attributes:
                nvars[nvar] = nvar

        if 'sentence_ends' in graph.metadata:
            sentence_ends = graph.metadata['sentence_ends'].split()
            sentence_ends = [int(x) for x in sentence_ends]
        else:
            sentence_ends = []
                
        return cls(tokens, nodes, edges, graph.top, penman=graph,alignments=alignments, sid=sid, nvars=nvars, sentence_ends=sentence_ends)

    def delete_node_leaving_no_trace(self, node_id):

        edges2delete = []
        
        for e in self.edges:
            if node_id in e:
                edges2delete.append(e)

        for e in edges2delete:
            if e in self.edges:
                self.edges.remove(e)

        del self.nodes[node_id]

    def check_connectivity(self):

        descendents = {n: {n} for n in self.nodes}
        edges2delete = []
        for x, r, y in self.edges:
            if x not in descendents or y not in descendents:
                print("will delete "+str((x, r, y)))
                edges2delete.append((x, r, y))
                continue
            descendents[x].update(descendents[y])
            for n in descendents:
                if x in descendents[n]:
                    descendents[n].update(descendents[x])
                    
        for e in edges2delete:
            if e in self.edges:
                self.edges.remove(e)

        to_be_deleted = []
        for nid in self.nodes:
            if len( descendents[nid] &  descendents[self.root] ) == 0 :
                print(self.nodes[nid])
                print("Nope, not connected, will be deleted")
                to_be_deleted.append(nid)
        for nid in to_be_deleted:
            self.delete_node_leaving_no_trace(nid)

        for e in self.edges:
            if e[0] not in self.nodes or e[2] not in self.nodes:
                print(e)
                print("Bad edge")

    def make_penman(self):

        self.un_invert() #normalizes all edges except for :coref-of
        
        all_tuples = []
        epidata = {}

        #add root edges first, thats how penman decides on 'top'
        for e in self.edges:
            if e[0] == self.root:
                all_tuples.append(e)
        
        for nid in self.nodes:
            if nid in self.nvars and self.nvars[nid] is not None:
                tup = (nid,":instance",self.nodes[nid])
                if tup not in all_tuples:
                    all_tuples.append(tup)
                    if self.alignments and nid in self.alignments and self.alignments[nid] is not None:
                        epidata[tup] = [surface.Alignment(indices=self.alignments[nid] if len(self.alignments[nid])<3 else [self.alignments[nid][0],self.alignments[nid][-1]])]
                        
        for e in self.edges:
            if self.nvars[e[0]] is None:
                print("\n!!!!! bad edge, constant as parent ... will not be printed "+ str(e))
                continue
            if self.nvars[e[2]] is None:
                if self.nodes[e[0]] == 'coref-entity':
                    print("\n!!!!! coref chains can not form over constants: "+ str(e))
                    continue
                #no node variable indicates constant valued attribute
                #FIXME adding explicit quotes to attributes
                tup = (e[0],e[1],f'"{self.nodes[e[2]]}"')
                if tup not in all_tuples:
                    all_tuples.append(tup)
                    if self.alignments and e[2] in self.alignments and self.alignments[e[2]] is not None:
                        epidata[tup] = [surface.Alignment(indices=self.alignments[e[2]] if len(self.alignments[e[2]])<3 else [self.alignments[e[2]][0],self.alignments[e[2]][-1]])]
            else:
                if e not in all_tuples:
                    all_tuples.append(e)
        self.penman = penman.graph.Graph(triples=all_tuples, epidata=epidata)
    
    def __str__(self,jamr=False):

        self.penman = None
        self.check_connectivity()
        self.make_penman()
        self.penman_str = penman.encode(self.penman) # self.to_penman(jamr=jamr)
        meta_data = ""
        if self.amr_id:
            meta_data  = '# ::id ' + self.amr_id + '\n'
        if self.doc_file:
            meta_data += '# ::doc_file ' + self.doc_file + '\n'
        if '::sentence_ends' not in meta_data and self.sentence_ends and len(self.sentence_ends)>0:
            snt_ends_str = [str(x) for x in self.sentence_ends]
            meta_data+= '# ::sentence_ends ' + ' '.join(snt_ends_str) + '\n'

        meta_data += '# ::tok ' + ' '.join(self.tokens) + '\n'
        #sanity check
        #p = penman.decode(self.penman_str)

        return meta_data + self.penman_str + '\n\n'


    #=====================================================================
    #the code below is old and it is for
    #1) adding corefs as node pair edges and
    #2) making such node pairs from coref chains

    def get_nodes_chains(self):

        node_corefs = {}
        for e in self.edges:
            if e[1] == AMR.coref_rel_inv:
                if e[0] in node_corefs and node_corefs[e[0]] != e[2]:
                    import ipdb; ipdb.set_trace()
                node_corefs[e[0]] = e[2]
            if e[1] == AMR.coref_rel:
                if e[2] in node_corefs and node_corefs[e[2]] != e[0]:
                    import ipdb; ipdb.set_trace()
                node_corefs[e[2]] = e[0]

        return node_corefs
        
    def move_bridges_to_chains(self):
            
        node_corefs = self.get_nodes_chains()

        if len(node_corefs) == 0:
            return
                
        edges2delete = []
        for (i,e) in enumerate(self.edges):
            if e[1] in [":subset-of",":part-of"]:
                n1 = e[0]
                if n1 in node_corefs:
                    n1 = node_corefs[n1]
                n2 = e[2]
                if n2 in node_corefs:
                    n2 = node_corefs[n2]
                edge = (n1,e[1],n2)
                if edge == e:
                    continue
                if edge not in self.edges:
                    self.edges[i] = (n1,e[1],n2)
                else:
                    edges2delete.append(e)
        for e in edges2delete:
            if e in self.edges:
                self.edges.remove(e)

    def merge_nodes_into_chain(self, node1, node2):
        
        if node1 not in self.nodes:
            print("cannot merge, node1 not found !!!")
            return
        if node2 not in self.nodes:
            print("cannot merge, node2 not found !!!")
            return

        node_corefs = self.get_nodes_chains()
        node1_corf = None
        if node1 in node_corefs:
            node1_corf = node_corefs[node1]
        node2_corf = None
        if node2 in node_corefs:
            node2_corf = node_corefs[node2]
        edges_to_delete = []
        
        for (i,e) in enumerate(self.edges):
            if e[0] == node1 and (e[2] == node2):
                    edges_to_delete.append(e)                    
            if e[0] == node2 and (e[2] == node1):
                    edges_to_delete.append(e)

        for e in edges_to_delete:
            if e in self.edges:
                self.edges.remove(e)

        if node1_corf is not None:
            coref_edge = (node2,AMR.coref_rel_inv,node1_corf)
            coref_edge_inv = (node1_corf,AMR.coref_rel,node2)
            if coref_edge not in self.edges and coref_edge_inv not in self.edges:
                self.edges.append(coref_edge)
            if node2_corf is not None and node2_corf != node1_corf:
                if node2_corf in self.nodes:
                    del self.nodes[node2_corf]
                    for (i,e) in enumerate(self.edges):
                        if e[0] == node2_corf:
                            new_edge = (node1_corf,e[1],e[2])
                            if new_edge not in self.edges:
                                self.edges[i] = new_edge
                            else:
                                edges_to_delete.append(e)
                        if e[2] == node2_corf:
                            new_edge = (e[0],e[1],node1_corf)
                            if new_edge not in self.edges:
                                self.edges[i] = new_edge
                            else:
                                edges_to_delete.append(e)
                else:
                    print("node2 coref not found !!!")
        elif node2_corf is not None:
            coref_edge = (node1, AMR.coref_rel_inv, node2_corf)
            self.edges.append(coref_edge)
        else:
            chain_nid = self.add_node(AMR.coref_node, prefix='cc')
            self.edges.append((node1,AMR.coref_rel_inv,chain_nid))
            self.edges.append((node2,AMR.coref_rel_inv,chain_nid))
            
        for e in edges_to_delete:
            if e in self.edges:
                self.edges.remove(e)
                
    def make_chains_from_pairs(self, coref_rel="same-as"):
        found = True
        while found:
            found = False
            for e in self.edges:
                if e[1] == ':'+coref_rel or e[1] == ':'+coref_rel+"-of":
                    found = True
                    if type(e[0]) == int or type(e[2]) == int:
                        self.edges.remove(e)
                    else:
                        self.merge_nodes_into_chain(e[0],e[2])                    
                    break

        self.move_bridges_to_chains()

    def add_edges(self, edges):
        for (s,l,t) in edges:
            if s in self.nvars2nodes and t in self.nvars2nodes:
                this_node = self.nvars2nodes[s]
                cref_node = self.nvars2nodes[t]
                if this_node not in self.nodes or cref_node not in self.nodes:
                    print("edge not added:")
                    print((s,l,t))
                    continue
                if l in [':part']:
                    edge = (this_node,l+"-of",cref_node)
                    if edge not in self.edges:
                        self.edges.append(edge)
                elif l == ':member':
                    edge = (this_node,":subset-of",cref_node)
                    if edge not in self.edges:
                        self.edges.append(edge)
                else:
                    edge = (this_node,l,cref_node)
                    if edge not in self.edges:
                        self.edges.append(edge)
            else:
                print("node varaiables not found for the edge: " + s + "\t" + t)

def process_corefs_into_triples(fnames):

    #convert coref chains into pairwise triple ... by linking all nodes to the first in the chain
    corefs = {}
    
    for fname in fnames:
        tree = ET.parse(fname)
        root = tree.getroot()
        doc_id = root[0].attrib['docid']
        doc_sen_ids = []
        order = {}
        for sen in root[0]:
            doc_sen_ids.append(sen.attrib['id'])
            order[sen.attrib['id']] = sen.attrib['order']
        document_triples = []
        chains_n_sngltns = {}

        #identity relations
        for i in range(len(root[1][0])):
            chain = []
            chain_id = root[1][0][i].attrib['relationid']
            for m in range(len(root[1][0][i])):
                sen_id = root[1][0][i][m].attrib['id']
                var = None
                if 'variable' in root[1][0][i][m].attrib:
                    var = root[1][0][i][m].attrib['variable']
                else:
                    var = root[1][0][i][m].attrib['parentvariable']
                rel = ':same-as'
                con = None
                if root[1][0][i][m].tag == 'implicitrole':
                    rel = ":"+root[1][0][i][m].attrib['argument']
                else:
                    con = root[1][0][i][m].attrib['concept']
                chain.append((int(order[sen_id]),sen_id,var,rel,con))
            (triples, chain_ref_node) = chain2triples(chain)
            chains_n_sngltns[chain_id] = chain_ref_node
            document_triples += triples

        #singltons
        for i in range(len(root[1][1])):
            sen_id = root[1][1][i][0].attrib['id']
            singleton_id = root[1][1][i].attrib['relationid']
            if 'variable' not in root[1][1][i][0].attrib.keys():
                print("implicite signleton ignored")
                continue
            var = root[1][1][i][0].attrib['variable']
            chains_n_sngltns[singleton_id] = sen_id +"." + var

        #bridges
        for i in range(len(root[1][2])):
            parent = ""
            parent_rel = ""
            kid_rel = ""
            kids = []
            for m in range(len(root[1][2][i])):
                rel_id = root[1][2][i][m].attrib['id']
                if rel_id not in chains_n_sngltns:
                    print('missing '+rel_id+' in '+fname)
                    continue
                ref_node = chains_n_sngltns[rel_id]
                tag = root[1][2][i][m].tag
                if tag in ['superset','whole']:
                    parent = ref_node
                    parent_rel = ":"+root[1][2][i][m].tag
                else:
                    kids.append(ref_node)
                    kid_rel = ":"+root[1][2][i][m].tag
            for kid in kids:
                if len(parent) == 0 :
                    print("missing parent")
                    break
                document_triples.append((kid, kid_rel, parent))

        corefs[doc_id] = (document_triples,doc_sen_ids,fname)

    return corefs
                
def chain2triples(chain):
    #chains of coref (sentence id, node variable, concept)

    triples = []
    sorted_chain = sorted(chain, key=lambda x: x[0])
    
    candidate_refs = []
    for ref in sorted_chain:
        if ref[3] == ':same-as':
            candidate_refs.append(ref)
            break #first explicite member in the chain will be main reference
    if len(candidate_refs) == 0:
        candidate_refs.append(sorted_chain[0])
    ref_node = None
    for ref in sorted_chain:
        node = str(ref[1])+"."+ref[2]
        rel = ref[3]
        if ref_node is not None:
            triples.append((node, rel, ref_node))
        if ref in candidate_refs:
            ref_node = node

    return (triples, ref_node)


