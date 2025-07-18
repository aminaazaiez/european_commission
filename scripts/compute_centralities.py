import networkx as nx
import xgi
import pandas as pd
import os
from collections import Counter
import json

from centrality import degree, cardinality, betweenness, hypercoreness
from core import core_decomposition
from utils import clique_expansion


from european_commission import EuropeanCommission


def compute_and_save(data_path: str, out_path: str):

    EC = EuropeanCommission(data_path, reload=False)    

    # Get sub-hypergraph without singleton edges

    def get_subhypergraph(X):
        H_orga = X.copy()
        # Remove EC members
        H_orga.remove_nodes_from(EC.entities[ EC.entities['Type'] == 'EC member'].index)
        # Keep the maximal connected component
        H_orga.remove_nodes_from(H_orga.nodes - xgi.largest_connected_component(H_orga))
        # Remove singleton edges
        e_remove = [edge for edge in H_orga.edges if H_orga.edges.size[edge] < 2]
        H_orga.remove_edges_from(e_remove)
        return H_orga
        
    H = EC.H.copy()
    H_orga = get_subhypergraph(H)
    G = clique_expansion(H)
    G_orga = get_subhypergraph(G)
    # Set clique expantion hypergraph

    # Set node and edge centrality dataframe
    node_cent_df = pd.DataFrame(index = pd.Index(list(H.nodes)).rename('Name'))
    edge_cent_df = pd.DataFrame(index = pd.Index(list(H.edges)).rename('Name'))

    for (X_H, X_G), suffix in zip([(H,G), (H_orga, G_orga)], ['','_sub']):
        for X, net_label in zip([X_H,X_G], ['', '_G']):
            # Degree
            cent = degree(X)
            label = [cent.columns[0]+net_label+suffix]
            node_cent_df.loc[list(X.nodes), label] = cent[cent.columns[0]]
            # Betweeness
            cent = betweenness(X)
            label = [cent.columns[0]+suffix]
            node_cent_df.loc[list(X.nodes), label] = cent[cent.columns[0]]
            if X == X_H:
                edge_cent_df.loc[list(X.edges), label] = cent[cent.columns[0]]

            # # EV centrality    
            # for mode in ['linear', 'log exp', 'max']:
            #     cent = eigenvector(X, mode)
            #     label = [cent.columns[0]+suffix]
            #     node_cent_df.loc[list(X.nodes), label] = cent[cent.columns[0]]
            #     if X == X_H:
            #         edge_cent_df.loc[list(X.edges), label] = cent[cent.columns[0]]  

            # Edge size    
            cent = cardinality(X)
            label = [cent.columns[0]+suffix]
            if X == X_H:
                edge_cent_df.loc[list(X.edges), label ] = cent[cent.columns[0]]  
                
            # Core decomposisition and Hypercoreness
            for multiedges in [True, False]:
                X.cleanup(multiedges = multiedges, relabel = False)
                core , core_e = core_decomposition(X, multiedges=multiedges)

                core = { int(m) : {int(k) :  [set(component) for component in core[m][k]] for k, value in core[m].items() } for m in core.keys()}
                core_e = { int(m) : {int(k) :  [set(component) for component in core_e[m][k]] for k, value in core_e[m].items() } for m in core_e.keys()}

                g_m = Counter( X.edges.size.asdict().values())
                g_m = {m : g_m.get(m, 0)/X.num_edges for m in range(2, int(max(g_m.keys())+1))}

                cent  = hypercoreness(core,g_m) 
                label = [f'{cent.columns[0]}{net_label+suffix}_{multiedges}']
                node_cent_df.loc[list(X.nodes) , label] = cent[cent.columns[0]]

                cent  = hypercoreness(core_e, g_m) 
                label = [f'{cent.columns[0]}{net_label+suffix}_{multiedges}']
                if X == X_H:
                    edge_cent_df.loc[list(X.edges) , label] = cent[cent.columns[0]]
                
                # Convert sets to lists before serialization
                def convert_sets_to_lists(obj):
                    if isinstance(obj, set):
                        return list(obj)
                    return obj
                with open(os.path.join(out_path, f"core{net_label+suffix}_multiedges{multiedges}.json"), "w") as outfile: 
                    json.dump(core, outfile, default=convert_sets_to_lists)
                with open(os.path.join( out_path, f"core_e{net_label+suffix}_multiedges{multiedges}.json"), "w") as outfile: 
                    json.dump(core_e, outfile, default=convert_sets_to_lists)


    ## Save results
    edge_cent_df.index = edge_cent_df.index.rename('Name')
    edge_cent_df.to_csv(os.path.join(out_path,'edge_centrality.csv'))
    node_cent_df.to_csv(os.path.join(out_path, 'node_centrality.csv'))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="compute centralities")
    p.add_argument("--data",  required=True, help="Path to your data folder")
    p.add_argument("--out",   required=True, help="Where to write centrality files")
    args = p.parse_args()
    compute_and_save(args.data, args.out)

#cd scripts python compute_centralities.py --data ../data --out ../out/metrics