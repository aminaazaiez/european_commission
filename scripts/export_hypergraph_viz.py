import networkx as nx
import xgi
import pandas as pd
import os

from european_commission import EuropeanCommission
from utils import XGI_2_nxBipartite 



def create_and_export(data_path: str, out_path: str):

    EC = EuropeanCommission(data_path, reload=False)    
    
    # Entire Hypergraph
    X = EC.H.copy()
    X.remove_nodes_from(EC.entities[ EC.entities['Category of registration'].isna()].index)
    B = XGI_2_nxBipartite(X)

    label = dict(zip( EC.entities[EC.entities['TR Name'].notna()].index ,EC.entities[EC.entities['TR Name'].notna()]['TR Name']))
    B = nx.relabel_nodes(B, label)

    attr = dict(zip(EC.entities['TR Name'], EC.entities['Category of registration']))
    attr.update( dict(zip(EC.entities.index, EC.entities['Category of registration'])))
    nx.set_node_attributes(B, attr,'Partition')

    attr = dict(zip(EC.entities['TR Name'], EC.entities['Type']))
    attr.update( dict(zip(EC.entities.index, EC.entities['Category of registration'])))
    nx.set_node_attributes(B, attr,'Type')

    nx.write_gexf(B, os.path.join(out_path, "H.gexf"))

    # Oganizations sub-hypergraph
    label = dict(zip(EC.entities.index, EC.entities['TR Name']))
    X = EC.H.copy()
    X.remove_nodes_from(EC.entities[ EC.entities['TR Name'].isna()].index)
    #X.cleanup(multiedges = False, connected = False , singletons  = True, relabel = False)
    B = XGI_2_nxBipartite(X)
    B = nx.relabel_nodes(B, label)
    attr = dict(zip(EC.entities['TR Name'], EC.entities['Category of registration']))
    #nx.set_node_attributes(B, partition_attr,'Partition')
    nx.set_node_attributes(B, attr,'Partition')

    nx.write_gexf(B, os.path.join(out_path, "orga.gexf"))

    print('proportion of nodes in the giant component', max([len(comp) for comp in xgi.connected_components(X) ]) / EC.H.num_nodes)
    ##

    # EC members sub-hypergraph
    attr = dict(zip(EC.entities.index , EC.entities['Category of registration']))
    X = EC.H.copy()
    X.remove_nodes_from(EC.entities[ EC.entities['Type'] == 'Organization'].index)
    B = XGI_2_nxBipartite(X)
    nx.set_node_attributes(B,attr,'Type')

    nx.write_gexf(B, os.path.join(out_path, "EC.gexf"))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Export EC hypergraphs to GEXF")
    p.add_argument("--data",  required=True, help="Path to your data folder")
    p.add_argument("--out",   required=True, help="Where to write GEXF files")
    args = p.parse_args()
    export_all(args.data, args.out)

#cd scripts python export_hypergraph_viz.py --data ../data/ --out ../out/graph_viz/