import pandas as pd
import itertools
import os
from collections import Counter
from collections.abc import Iterable

from transparency_register import TransparencyRegister
from orbis import Orbis

def create_entities_df(meetings: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame of all EC members and registered organizations."""
    ec_members    = set(itertools.chain.from_iterable(meetings['EC member']))
    organizations = set(itertools.chain.from_iterable(meetings['TR ID']))

    df_org = pd.DataFrame(
        {'Type': ['Organization'] * len(organizations)},
        index=list(organizations)
    )
    df_ec = pd.DataFrame(
        {'Type': ['EC member'] * len(ec_members)},
        index=list(ec_members)
    )
    entities = pd.concat([df_org, df_ec])
    entities.index.rename('Name', inplace=True)
    return entities


def mapper_duplicates(df: pd.DataFrame, duplicated_key: str, mapper: dict) -> None:
    """
    Update mapper in-place to collapse any rows with duplicated_key into a single reference.
    `mapper` is modified so that any duplicate index maps to one “reference” index.
    """
    df = df.dropna(subset = duplicated_key)[df.dropna(subset = duplicated_key).duplicated(duplicated_key, keep = False)].copy()

    for name in df[duplicated_key]:
        indices_of_duplicated = list(df[df[duplicated_key] == name].index)

        ref_key = set(indices_of_duplicated) & set(mapper.values())
        if len(ref_key) > 1:
            print('Both keys are reference keys in mapper')
            
        elif len(ref_key) == 0:
            ref_key = indices_of_duplicated[0]
            other_keys = indices_of_duplicated[1:]

        else :
            other_keys = list(set(indices_of_duplicated) - ref_key)
            ref_key = list(ref_key)[0]

        mapper.update(dict( zip(other_keys , [ref_key]*len(other_keys)) ))

    



def members_info(meetings: pd.DataFrame, out_path: str, reload: bool=False) -> pd.DataFrame:
    """
    Build (or reload) a table of EC-member names, titles and department.
    Saves to/loads from `{out_path}/EC_members_info.csv`.
    """
    csv_file = f"{out_path}/EC_members_info.csv"
    if reload or not os.path.exists(csv_file):
        rows = []
        for ec_tuple, title_tuple, dept in zip(
            meetings['EC member'],
            meetings['Title of EC representative'],
            meetings['Department']
        ):
            for name, title in zip(ec_tuple, title_tuple):
                rows.append({'Name': name, 'Title': title, 'Department': dept})
        df = pd.DataFrame(rows).drop_duplicates()
        df.to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
    df.index = df['Name']
    return df


def add_entities_info(entities: pd.DataFrame,
                      data_path: str,
                      members_df: pd.DataFrame,
                      mapper: dict,
                      reload: bool=False) -> pd.DataFrame:
    """
    Enrich `entities` with:
      - Category of registration (from EC Title)
      - Transparency-Register info
      - Orbis info (with duplicate collapse)
    """
    # 1) EC member categories
    titles_map = {
        'Cabinet member': 'Cabinet member',
        'Commissioner':    'Commissioner',
        'Vice-President':  'Commissioner',
        'President':       'Commissioner',
        'Executive Vice-President': 'Commissioner',
        'High Representative':      'Commissioner',
        'Director-General': 'Director-General',
        'Head of service':  'Director-General',
        'Acting Director-General': 'Director-General',
        'Secretary-General':        'Director-General',
        'Acting Head of service':   'Director-General',
        'Director of Office':       'Director-General'
    }
    members_df['Category of registration'] = members_df['Title'].map(titles_map)
    # keep only members we have in entities
    members_df = members_df.loc[members_df.index.intersection(entities.index)]  

    entities = entities.copy()
    entities.loc[members_df.index, 'Category of registration'] = members_df['Category of registration']

    # Transparency Register data
    TR = TransparencyRegister(data_path, reload=reload)
    tr_cols = ['TR Name','TR Country','Category of registration','Level of interest','Fields of interest','Members FTE']
    TR = TR.loc[TR.index.intersection(entities.index)]
    entities.loc[TR.index, tr_cols] = TR[tr_cols]
    entities['Category of registration'] = entities['Category of registration'].str.replace('&','and')

    mapper_duplicates(entities, 'TR Name', mapper)

    # Orbis data
    orb = Orbis(f"{data_path}/Orbis/")
    orb_matched = orb.matched_names.copy()
    orb_matched = orb_matched.loc[entities.index.intersection(orb_matched.index)]
    mapper_duplicates(orb_matched, 'BvD ID', mapper)
    # collapse and rebuild hypergraph at top level after this
    orb_matched['TR ID'] = orb_matched.index

    df = orb_matched.merge(
        orb.company_data,
        on='BvD ID',
        how='left'
    )
    df.index = df['TR ID']
    orbis_cols = [
        'Orbis Country','Country ISO','BvD sectors','Revenue',
        'Assets','Nb employees','NACE','corporate group size','Entity type',
        'GUO Name','GUO Country ISO','GUO Type','GUO NACE','GUO Revenue','GUO Assets','GUO Nb employees'
    ]
    entities.loc[df.index, orbis_cols] = df[orbis_cols]

    return entities

