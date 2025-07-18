from collections import Counter
from bisect import bisect_right

import pandas as pd
import xgi

def rich_club(edge_list, degree_dict=None):
    """

    """
    degree_dict = _compute_degrees(edge_list)

    # Precompute min degrees for edges
    min_degrees = [min(degree_dict[node] for node in edge) for edge in edge_list]
    max_k = max(min_degrees)
    sorted_min_degrees = sorted(min_degrees)

    rc = {}

    sorted_min_degrees = sorted(min_degrees)
    for k in range(1, max_k):
        # Binary search to find how many min_degrees are > k
        from bisect import bisect_right
        num_edges = len(min_degrees) - bisect_right(sorted_min_degrees, k)
        sum_deg = sum(deg for deg in degree_dict.values() if deg > k) 
        rc[k] = num_edges 
        rc[k] = num_edges / sum_deg if sum_deg > 0 else 0.0

    return rc

def normalized_rich_club(df_sim, mode, multiedges, EC):

    EC.meetings['TR ID'] = EC.meetings['TR ID'].apply(lambda x: tuple(sorted(list(set(x)))))
    EC.meetings['EC member'] = EC.meetings['EC member'].apply(lambda x: tuple(sorted(list(set(x)))))

    if multiedges:
        edges_df = EC.meetings.copy()
    else:
        edges_df = EC.meetings.drop_duplicates(subset = ['TR ID', 'EC member'])

    if mode == 'full' or 'subdim':
        cleanup_kwargs = {    
            'isolates' : False,
            'singletons': False,
            'multiedges': multiedges,
            'connected': False, # To test
            'relabel': False,
            'in_place': True}
        # Empirical Hypergraph
        edges_emp = edges_df.apply(lambda x: x['TR ID'] + x['EC member'], axis = 1)
        H = xgi.Hypergraph(list(edges_emp))
        H.cleanup(**cleanup_kwargs)

    elif mode == 'orga':
        # Empirical orga
        H = xgi.Hypergraph(list(edges_df.apply(lambda x: x['TR ID'], axis = 1))) # Ignore duplicated nodes 
        H.cleanup(**cleanup_kwargs)


    # Compute empirical rich-club
    rc_emp = pd.Series(rich_club(H.edges.members()))

    # Compute rich-club from simulations
    df_group = df_sim[(df_sim['mode'] == mode) & (df_sim['multiedges'] == multiedges)]
    rc_rand_list = [pd.Series(rich_club(Hsim.edges.members())) for Hsim in df_group['H']]

    # Align lengths (fill shorter ones with NaNs)
    df_rand = pd.concat(rc_rand_list, axis=1).T
    df_rand = df_rand.fillna(0)

    mean_rand = df_rand.mean()
    std_rand = df_rand.std()

    # Compute normalized coefficient
    rho = rc_emp / mean_rand
    rho_err = (rc_emp / mean_rand**2) * (std_rand) /(len(df_rand)**0.5)
    rho_std = (rc_emp / mean_rand**2) * (std_rand) 

    results = {
        'mode': mode,
        'multiedges': multiedges,
        'rho': rho,
        'err': rho_err,
        'std': rho_std,
        'mean': mean_rand 
    }
    return results

def _compute_degrees(edge_list):
    degree = Counter()
    for edge in edge_list:
        for node in edge:
            degree[node] += 1
    return degree