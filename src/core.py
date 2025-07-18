import xgi
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm
from collections import Counter

def restrict_m_hypergraph(X : xgi.Hypergraph, m: int, multiedges ):
    ''' m : minimum size of hyperedge'''
    if not multiedges:
        X.merge_duplicate_edges()
    e_remove = [edge for edge in X.edges if X.edges.size[edge] < m]
    X.remove_edges_from(e_remove)
    still_2_remove = len(e_remove) > 0
    return(X, still_2_remove)

def restrict_k_hypergraph(X : xgi.Hypergraph, k: int, multiedges):
    ''' k : minimum degree of nodes'''
    if not multiedges:
        X.merge_duplicate_edges()
    n_remove = [node for node in X.degree().keys() if X.degree()[node]< k]
    X.remove_nodes_from(n_remove)
    still_2_remove = len(n_remove) > 0
    return(X , still_2_remove)

def core_decomposition(H : xgi.Hypergraph, multiedges) :
    X = H.copy()
    M = range(2 , max(X.edges.size.asnumpy()) + 1 ) # m = edge size
    core = { m : {} for m in M }
    core_e = { m : {} for m in M}

    for m in tqdm(M): # loop for the m shells
        # For each m, start with the initial hypergraph restricted to the edges of size >= m
        k = 1
        X = H.copy()
        while X.num_nodes > 0 : # loop for the k,m shell
            X , still_edges_2_remove = restrict_m_hypergraph(X, m, multiedges)
            X , still_nodes_2_remove = restrict_k_hypergraph(X, k, multiedges)
            # Store previous shell to compute the k,m shell at the end of the loop

            while  still_nodes_2_remove or still_edges_2_remove : # redo untill there are neither nodes nore edges that can be removed
                X , still_nodes_2_remove = restrict_m_hypergraph(X, m, multiedges)
                X , still_edges_2_remove = restrict_k_hypergraph(X, k, multiedges)

            if X.num_nodes > 0 :
                core[m][k] = [list(component) for component in xgi.connected_components(X)]
                core_e[m][k] = [list(X.edges)]
                k += 1
    return core, core_e

def k_m_core(H : xgi.Hypergraph, m,k, multiedges):
    X = H.copy()
    X , still_nodes_2_remove = restrict_m_hypergraph(X, m, multiedges)
    X , still_edges_2_remove = restrict_k_hypergraph(X, k, multiedges)
    while  still_nodes_2_remove or still_edges_2_remove : # redo untill there are nither nodes nore edges that can be removed
            X , still_nodes_2_remove = restrict_m_hypergraph(X, m, multiedges)
            X , still_edges_2_remove = restrict_k_hypergraph(X, k, multiedges)
    return(X)


def _k_m_largest_cc_size(core,m , k):
    components = core[m][k]
    if len(components)>1:
        components.sort(key = len, reverse = True)
        return(len(components[0]) /( len(components[0])+ len(components[1]) ) )
    else :
        return(1)

def proportion_largest_cc_size(core):
    M = sorted(core.keys())
    K = sorted({k for m_dict in core.values() for k in m_dict})
    array = np.full((len(M), len(K)), np.nan)

    # Populate the array
    for i, m in enumerate(M):
        for j, k in enumerate(K):
            if k in core[m]:  
                array[i, j] = _k_m_largest_cc_size(core, m, k)

    return array, M, K


def k_m_core_size(core):
    M = sorted(core.keys())
    K = sorted({k for m_dict in core.values() for k in m_dict})
    # Initialize an array to store the core sizes
    n_k_m = np.zeros((len(M), len(K)))
    # Populate the array
    for i, m in enumerate(M):
        for j, k in enumerate(K):
            if k in core[m]:
                n_k_m[i, j] = sum(len(component) for component in core[m][k])
    return n_k_m, M, K



def species_survival(core, m , k , node_set):
    k_m_core = core[m][k]
    nodes = set()
    for component in k_m_core:
        nodes = nodes.union(component)

    return( len( nodes & node_set ) )



def exiting_shell(core) :
    x_min = min(core.keys())
    y_min = min(core[x_min].keys())
    c_m = { node : {} for node in core[x_min][y_min][0]}
    for x in core.keys():
        y_max = max(core[x].keys())
        for y in core[x].keys():
            if y != y_max :
                x_y_shell = core[x][y][0] - core[x][y+1][0]
            else :
                x_y_shell = core[x][y][0]

            for node in x_y_shell:
                c_m[node][x] = y
    return(c_m)

