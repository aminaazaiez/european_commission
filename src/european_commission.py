import pandas as pd
import os
import itertools

from collections import Counter
from collections.abc import Sequence, Iterable

import xgi

from process_meetings import (
    load_meetings, 
    add_relative_commission,
    merge_duplicated_meetings,
    collapse_entities)
from process_entities import (
    create_entities_df,
    members_info,
    add_entities_info,
)

from utils import clique_expansion

date_start = '2019-12-01'
date_end = '2024-11-30'



class EuropeanCommission:
    def __init__(self, data_path: str, reload: bool=False, fuzzy_ratio: int =70):
        self.data_path = data_path
        self.reload    = reload
        self.mapper    = {}

        if reload:
            # load and process meetings
            m = load_meetings(data_path, date_start, date_end, fuzzy_ratio)
            m.rename(columns={
                'Name of EC representative':    'EC member',
                'Date of meeting':               'Date',
                'Transparency register ID':      'TR ID',
                'Name of DG - full name':        'Name of DG',
                'Subject of the meeting':        'Subject'
            }, inplace=True)
            m['Date'] = pd.to_datetime(m['Date'])
            for col in ['EC member','Title of EC representative','TR ID']:
                m[col] = m[col].str.split(',').apply(tuple)

            m = add_relative_commission(m, data_path)
            m = merge_duplicated_meetings(m)
            # save processed meetings into json file
            m.to_json(os.path.join(data_path, 'meetings.json'))
            # init self.meetings
            self.meetings = m
        else:
            # load processed meetings file
            m = pd.read_json(os.path.join(data_path, 'meetings.json'))
            for col in ['EC member','Title of EC representative','TR ID','Department']:
                m[col] = m[col].apply(tuple)
            self.meetings = m

        # Create entities dataframe
        self.entities = create_entities_df(self.meetings)
        self.generate_hypergraph()
        # Add ec members titles and departments using meetings df
        mems = members_info(self.meetings, data_path, reload=reload) 
        # Add entities info with processing for duplicated entities
        self.entities = add_entities_info(self.entities, data_path, mems, self.mapper, reload=reload)

        # Generate hypergraph
        self.meetings = collapse_entities(self.meetings, self.mapper)
        self.generate_hypergraph()
        
    def generate_hypergraph(self):
        hyperedges =  self.meetings['EC member'] + self.meetings['TR ID']

        # Select the maximal connected subgraph

        self.H = xgi.Hypergraph( dict( hyperedges.apply(lambda x : list(x))))
        #self.H.remove_nodes_from(self.H.nodes - xgi.largest_connected_component(self.H))

        self.entities = self.entities.loc[list(self.H.nodes)]
        self.meetings = self.meetings.loc[list(self.H.edges)]
 

    def sub_hypergraph(self, kind = 'Orga'):
        H =  self.H.copy()
        if kind == 'Orga':
            H.remove_nodes_from(self.entities[ self.entities['Type'] == 'EC member'].index)
            return H
        elif kind == 'EC':
            H.remove_nodes_from(self.entities[ self.entities['Type'] == 'Organization'].index)
            return H


    def get_orga(self):
        return(self.entities[self.entities['Type'] == 'Organization'])

    def get_companies(self):
        return(self.entities[self.entities['Category of registration'] == 'Companies and groups'])


    def save_orga_names_batchs(self, columns : list, path_file : str):
        df = self.get_companies()[columns]
        df.index.rename('TR ID' , inplace = True)
        rows_per_file = 100

        # Calculate the total number of files needed
        total_files = len(df) // rows_per_file + (len(df) % rows_per_file > 0)

        # Split the DataFrame into smaller chunks and save each chunk as a separate file
        for i in range(total_files):
            start_idx = i * rows_per_file
            end_idx = start_idx + rows_per_file
            chunk_df = df.iloc[start_idx:end_idx]
            chunk_df.to_csv(path_file + f'company_names_{i + 1}.csv',  sep = '\t', index = True)


