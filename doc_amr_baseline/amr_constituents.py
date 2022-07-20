from collections import defaultdict
from ipdb import set_trace
import re


def get_subgraph_by_id(amr, no_reentrancies=True, no_reverse_edges=False):
    '''
    Given an AMR class provide for each node a list of all nodes "below" it
    in the graph. Ignore re-entrancies and reverse edges if solicited. Not
    ignoring re-entrancies can lead to infinite loops
    '''

    # get re-entrant edges
    if no_reentrancies:
        reentrancy_edges = get_reentrancy_edges(amr)
    else:
        reentrancy_edges = []

    # Gather constituents bottom-up
    # find leaf nodes
    leaf_nodes = []
    for nid, nname in amr.nodes.items():
        child_edges = [(nid, label, tgt) for tgt, label in amr.children(nid)]
        # If no nodes, or nodes are re-entrant
        if set(child_edges) <= set(reentrancy_edges):
            leaf_nodes.append(nid)
    # start from leaf nodes and go upwards ignoring re-entrant edges
    # store subgraph for every node as all the nodes "below" it on the tree
    subgraph_by_id = defaultdict(set)
    candidates = leaf_nodes
    new_nodes = True
    count = 0
    while new_nodes:
        new_candidates = set()
        new_nodes = False
        for nid in candidates:
            # ignore re-entrant nodes
            unique_parents = []
            for (src, label) in amr.parents(nid):
                if (
                    (src, label, nid) not in reentrancy_edges
                    and not (no_reverse_edges and label.endswith('-of'))
                ):
                    unique_parents.append(src)
            if len(unique_parents) == 0:
                continue
            elif len(unique_parents) > 1:
                set_trace(context=30)
            # colect subgraph for this node
            src = unique_parents[0]
            subgraph_by_id[src] |= set([nid])
            subgraph_by_id[src] |= set(subgraph_by_id[nid])
            new_candidates |= set([src])
            new_nodes = True

        candidates = new_candidates

        count += 1
        if count > 1000:
            set_trace(context=30)
            print()

    return subgraph_by_id


def get_constituents_from_subgraph(amr):
    '''Get spans associated to each subgraph'''

    # get the subgraph below each node
    subgraph_by_id = get_subgraph_by_id(amr)

    def get_constituent(nid):
        '''Given nid and subgraph extract span aligned to it'''
        # Token aligned to node
        indices = amr.alignments[nid]
        sids = subgraph_by_id[nid]
        if sids is not None:
            # Tokens aligned to all nodes below it
            for id in sids:
                if amr.alignments[id] is None:
                    continue
                for idx in amr.alignments[id]:
                    if indices is None:
                        print('Alignment for node ',nid, 'not found but constituents are added')
                        indices = []
                    indices.append(idx)
        if indices:
            return min(indices), max(indices)
        else:
            return None, None

    # gather constituents associated to each node
    candidates = [amr.root]
    depth = 0
    depths = [depth]
    constituent_spans = []
    count = 0
    while candidates and count < 1000:
        nid = candidates.pop()
        ndepth = depths.pop()
        start, end = get_constituent(nid)
        if start is None:
            count += 1
            continue
        # Add constituent to list
        constituent_spans.append(dict(
            depth=ndepth,
            indices=(start, end+1),
            head=amr.nodes[nid],
            head_position=amr.alignments[nid],
            nid=nid
        ))
        reentrancy_edges = get_reentrancy_edges(amr)
        # update candidates, ignore re-entrant nodes
        candidates.extend([
            tgt for tgt, label in amr.children(nid)
            if (nid, label, tgt) not in reentrancy_edges
        ])
        depth += 1
        depths.extend([depth for _ in range(len(amr.children(nid)))])
        count += 1

    if count == 1000:
        # We got trapped in a loop
        set_trace(context=30)
        pass

    return {'tokens': amr.tokens, 'constituents': constituent_spans}


def get_reentrancy_edges(amr):

    # Get re-entrant edges i.e. extra parents. We keep the edge closest to
    # root
    # annotate depth at which edeg occurs
    candidates = [amr.root]
    depths = [0]
    depth_by_edge = dict()
    while candidates:
        for (tgt, label) in amr.children(candidates[0]):
            edge = (candidates[0], label, tgt)
            if edge in depth_by_edge:
                continue
            depth_by_edge[edge] = depths[0]
            candidates.append(tgt)
            depths.append(depths[0] + 1)
        candidates.pop(0)
        depths.pop(0)

    # in case of multiple parents keep the one closest to the root
    reentrancy_edges = []
    for nid, nname in amr.nodes.items():
        parents = [(src, label, nid) for src, label in amr.parents(nid)]
        if nid == amr.root:
            # Root can not have parents
            reentrancy_edges.extend(parents)
        elif len(parents) > 1:
            # Keep only highest edge from re-entrant ones
            # FIXME: Unclear why depth is missing sometimes
            reentrancy_edges.extend(
                sorted(parents, key=lambda e: depth_by_edge.get(e, 1000))[1:]
            )
    return reentrancy_edges

def get_predicates(amr):
    pred_regex = re.compile('.+-[0-9]+$')

    
    num_preds = 0
    pred_ret = []
    
    predicates = {n:v for n,v in amr.nodes.items() if pred_regex.match(v) and not v.endswith('91') and v not in ['have-half-life.01']}
    num_preds += len(predicates)
    for pred in predicates:
        if amr.alignments[pred] is not None:

            args = {
                        trip[1][1:].replace('-of', ''):(amr.nodes[trip[2]],amr.tokens[min(amr.alignments[trip[2]])],amr.alignments[trip[2]])
                        for trip in amr.edges
                        if trip[0] == pred #and trip[1].startswith(':ARG')
            }
        
        
            pred_ret.append({'pred':predicates[pred],'text':amr.tokens[min(amr.alignments[pred])],'args':args,'beg':min(amr.alignments[pred]),'end':max(amr.alignments[pred])+1})
            
    return pred_ret  

