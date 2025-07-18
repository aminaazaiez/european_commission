from sknetwork.clustering import*
from sknetwork.data import from_edge_list


def xgi_2_sknetwork(H):
    edges_id ,  edges  = X._edge.keys() , X._edge.values()

    edge_list = []

    for i , edge  in zip( edges_id , edges ):
        for node in edge:
            edge_list.append(( node, edge))

    graph = from_edge_list(edge_list, bipartite=True, sum_duplicates =True)
    return graph

def array2dict(A, node_labels):
    ''' Using a 1-D array where A[i] is the cluster to which agent i belongs, return the dictionnary of partition
        Paramters
        ---------
        A : 1D array shape (nb_nodes,)

        node_labels: 1D array shape (nb_nodes,)

        Returns
        -------
        clusters_: dict
            clusters_[i]: list of agent belonging to the cluster i'''
    n= len(set(A)) # = number of clusters
    # clusters of nodes
    clusters_ ={i : [] for i in range(n)}
    for c , p in zip( A, node_labels):
        clusters_[c].append(p)
    return(clusters_)


def part2dict(A):
    """
    Given a partition (list of sets), returns a dictionary mapping the part for each vertex
    """
    x = []
    for i in range(len(A)):
        x.extend([(a, i) for a in A[i]])
    return {k: v for k, v in x}



network =  xgi_2_sknetwork(EC.H)

adjacency_matrix = network.biadjacency

louvain = Louvain(resolution = 0.2, modularity ='Newman' , shuffle_nodes = True, random_state = 10)

louvain.fit(adjacency_matrix)
clustering = louvain.labels_row_

names = [EC.entities['TR Name'].loc[id] for id in network.names ]

clusters = array2dict(clustering, names)


partition_attr =  part2dict(clusters)
