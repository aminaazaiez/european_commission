import random
from itertools import combinations
import pandas as pd
import numpy as np




def EC_Orga_assortativity(meetings, degree_dict ):

    EC_member = meetings['EC member'].apply(lambda x : np.random.choice( x))
    Orga = meetings['TR ID'].apply(lambda x : np.random.choice( x))
    df = pd.concat([EC_member, Orga], axis = 1)
    df['k_ec'] = df['EC member'].apply(lambda x : degree_dict[x])
    df['k_orga'] = df['TR ID'].apply(lambda x : degree_dict[x])
    return df[['k_ec', 'k_orga']].corr('spearman')['k_ec']['k_orga']


def random_choice(x):
    if len(x)>1:
        return  np.random.choice(x, 2, False)
    else:
        return np.nan

def overall_assortativity(meetings, degree_dict):

    df = pd.DataFrame( (meetings['EC member'] + meetings['TR ID']).rename('edge'))
    df['u,v'] = df['edge'].apply(lambda x : np.random.choice(x, 2, False) )
    df['k_u'] = df['u,v'].apply(lambda x : degree_dict[x[0]])
    df['k_v'] = df['u,v'].apply(lambda x : degree_dict[x[1]])
    return df[['k_u', 'k_v']].corr('spearman')['k_u']['k_v']

def intra_assortativity(meetings, cat, degree_dict):

    df = pd.DataFrame( (meetings[cat]).rename('edge'))
    df['u,v'] = df['edge'].apply(lambda x : random_choice(x) )
    df.dropna(inplace = True)
    df['k_u'] = df['u,v'].apply(lambda x : degree_dict[x[0]])
    df['k_v'] = df['u,v'].apply(lambda x : degree_dict[x[1]])
    return df[['k_u', 'k_v']].corr('spearman')['k_u']['k_v']


