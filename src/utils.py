
import pandas as pd
import itertools
from typing import Iterable
import xgi
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

def split_column(df : pd.DataFrame, column : str, sep :str =', ' ,  prefix : str = None):
    items = set(itertools.chain.from_iterable([item.split(sep) for item in df[column].dropna().unique()]))
    df = pd.DataFrame( df[column])

    for item in items:
        if prefix :
            col_name = prefix + ' ' + item
        else :
            col_name = item
        # Initialize new columns with 0
        df[col_name] = pd.Series()
        df[col_name] = df[col_name].where(df[column].isna(), 0)
        # Use .str.contains() to set the values to 1 where applicable
        df.loc[ df[column].str.contains(item, case=False , regex = True) == True, col_name] = 1 # to deal with nan


    df.drop( column , axis = 1 , inplace = True)

    #df = df.astype('int')
    return(df)



# def mask_by_str(df : pd.DataFrame, column :str, value : str):
#     return(df.dropna(subset = column)[EC.entities.dropna(subset = column)[column].str.contains(value)])

def search_str_in_column_of_tuple(dataframe : pd.DataFrame, column_name : str, search_string : str):
    filter_condition = dataframe[column_name].apply(lambda x: any(search_string in element for element in x))
    filtered_df = dataframe[filter_condition]
    return filtered_df

def replace_element_in_tuple(x,mapper):
    return( tuple(mapper.get(item, item) for item in x))


def merge_iterable_of_tuples( l1 : Iterable , l2 : Iterable):
    return( [ tuple_1 + tuple_2 for tuple_1 , tuple_2 in zip ( l1 , l2) ] )


def hyperedges2biartite( edges_id ,  edges ):
    edge_name, node_name = [] ,[]

    for i , edge  in zip( edges_id , edges ):
        for node in edge:
            edge_name.append(i)
            node_name.append(node)
    return(pd.DataFrame( {'edge' : edge_name , 'node' : node_name}))


def XGI_2_nxBipartite (H : xgi.Hypergraph) :
    B = nx.Graph()

    B.add_nodes_from(H.nodes, bipartite='entity')
    B.add_nodes_from(H.edges , bipartite='meeting')
    for node in H.nodes:
        for edge in H.nodes.memberships(node):
            B.add_edge(node, edge)
    return B


def clique_expansion(X):
    """
    Return the full clique (2-section) expansion of an xgi.Hypergraph,
    keeping *all* multiplicities: every time two nodes co-occur in an
    original hyperedge, a size-2 edge is added to the output.

    Parameters
    ----------
    H : xgi.Hypergraph

    Returns
    -------
    G : xgi.Hypergraph
        Pairwise projection with multiedges preserved.
    """
    pair_edges = []
    H = X.copy()
    for e in H.edges:
        nodes = H.edges.members(e)
        # skip singletons
        if len(nodes) < 2:
            continue
        # add one 2-node edge for every unordered pair in the hyperedge
        pair_edges.extend(frozenset(pair) for pair in itertools.combinations(nodes, 2))

    return xgi.Hypergraph(pair_edges)