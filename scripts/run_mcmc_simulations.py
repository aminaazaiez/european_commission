import os
import random
import argparse
import multiprocessing as mp
import joblib
import numpy as np
import xgi
from tqdm.auto import tqdm

from european_commission import EuropeanCommission
from config_mcmc import vertex_labeled_MH, vertex_labeled_MH_subdim

def run_and_save_H(param):
    edges_df, mode, multiedges, n_steps, seed, nb_sim, out_dir = param
    if not multiedges:
        edges_df = edges_df.drop_duplicates(subset = ['TR ID', 'EC member'])

    # Prepare kwargs
    cleanup_kwargs = {    
        'isolates' : False,
        'singletons': False,
        'multiedges': multiedges,
        'connected': False, # To test
        'relabel': False,
        'in_place': True}

    if mode == 'full':
        # Empirical Hypergraph
        edges_emp = edges_df.apply(lambda x: x['TR ID'] + x['EC member'], axis = 1) # Ignore duplicated nodes
        H = xgi.Hypergraph(list(edges_emp))
        H.cleanup(**cleanup_kwargs)
        edges_input = H.edges.members()

    elif mode == 'subdim':
        # Empirical subdim
        H_orga = xgi.Hypergraph(list(edges_df.apply(lambda x: x['TR ID'], axis = 1))) # Ignore duplicated nodes 
        H_ec = xgi.Hypergraph(list(edges_df.apply(lambda x: x['EC member'], axis = 1)))
        edges_input = [list( [list(H_orga.edges.members(e)), list(H_ec.edges.members(e))]) for e in H_ec.edges ]
        
        edges_flat = [np.concatenate([list(e1), list(e2)]) for e1, e2 in edges_input]
        H = xgi.Hypergraph(edges_flat)

    elif mode == 'orga':
        # Empirical orga
        H = xgi.Hypergraph(list(edges_df.apply(lambda x: x['TR ID'], axis = 1))) # Ignore duplicated nodes 
        H.cleanup(**cleanup_kwargs)
        edges_input = H.edges.members()# remove singleton edges
    
    MCMC_kwargs = {
                    'edges': edges_input,
                    'mode': mode,
                    'n_steps': n_steps,
                    'burnin_steps': 2 * n_steps,
                    'multiedges': multiedges,
                    'n_clash': 0,
                    'seed': seed,
                    'sim': nb_sim
                    }

    # Run
    if mode == 'subdim':
        edges_sub = vertex_labeled_MH_subdim(**MCMC_kwargs)
        edges = [np.concatenate([list(e1), list(e2)]) for e1, e2 in edges_sub]
        H = xgi.Hypergraph(edges)
        
    else:
        H = xgi.Hypergraph(vertex_labeled_MH(**MCMC_kwargs))

    # Save
    fname = f"mode_{mode}_multi_{multiedges}_n_{n_steps}_sim_{nb_sim}.joblib"
    filepath = os.path.join(out_dir, fname)

    param = {'H': H, 'mode': mode, 'multiedges': multiedges, 'nsteps': n_steps, 'sim': nb_sim}
    joblib.dump(param, filepath)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True)
    parser.add_argument("--out",    required=True)
    args = parser.parse_args()

    EC = EuropeanCommission(data_path=args.data, reload=False)
    EC.meetings['TR ID'] = EC.meetings['TR ID'].apply(lambda x: tuple(sorted(list(set(x)))))
    EC.meetings['EC member'] = EC.meetings['EC member'].apply(lambda x: tuple(sorted(list(set(x)))))


    edges_df = EC.meetings.copy()
    n_steps = 15_000


    param_list = [
        (edges_df, mode, multiedges, n_steps, random.randint(0, 10**6), i, args.out)
        for mode in ['full','subdim', 'orga']
        for multiedges in [True, False]
        for i in range(100)
    ]

    with mp.Pool() as pool:
        pool = mp.Pool()
        pool.map(run_and_save_H, param_list)
        pool.close()