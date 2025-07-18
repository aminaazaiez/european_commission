import pandas as pd
import itertools

from fuzzywuzzy import fuzz
from unidecode import unidecode

def load_meetings(data_path, date_start, date_end, fuzzy_ratio = 70):

    # Load cabinet meetings

    file_name = 'Meetings of Commission representatives of the Von der Leyen Commission (2019-2024).xlsx'
    columns = ['Name of cabinet', 'Name of EC representative', 'Title of EC representative', 'Transparency register ID', 'Name of interest representative', 'Date of meeting', 'Location','Subject of the meeting']
    meetings = pd.read_excel(data_path + 'meetings/' + file_name, skiprows=[0], usecols=columns )


    # Load DG meetings

    file_name = 'Meetings of Directors-General of the European Commission.xlsx'
    columns = ['Name of DG - full name', 'Name of EC representative', 'Title of EC representative', 'Transparency register ID', 'Name of interest representative','Date of meeting', 'Location', 'Subject of the meeting']
    meetings = pd.concat([meetings , pd.read_excel(data_path + 'meetings/' + file_name, skiprows=[0], usecols=columns ) ] ,  ignore_index= True)

    to_drop = ['Recovery and Resilience Task force', 'Regulatory Scrutiny Board' , 'Informatics', 'European Personnel Selection Office', 'Task Force for Relations with the United Kingdom']

    idx = meetings.loc[meetings['Name of DG - full name'].isin(to_drop)].index
    meetings.drop(index = idx, inplace  = True)

    # Filter by date

    meetings = meetings[(meetings['Date of meeting'] > date_start) & (meetings['Date of meeting'] < date_end)]

    # Process Subject column
    meetings = preprocess_df(meetings, 'Subject of the meeting')

    def fuzzy_matching(group, fuzzy_ratio=fuzzy_ratio):
        # Generate combinations of subjects to compare
        for subj1, subj2 in itertools.combinations(group, 2):
            ratio = fuzz.partial_ratio(subj1, subj2)
            # If fuzzy ratio >= fuzzy_ratio, consider them as the same subject
            if ratio >= fuzzy_ratio:
                group = group.str.replace(subj2, subj1)
        return group

    df = meetings.groupby(['Date of meeting', 'Transparency register ID'])['Subject of the meeting'].apply(fuzzy_matching).reset_index()
    df.index = df['level_2']
    meetings['Subject of the meeting'] = df['Subject of the meeting']
    return meetings


def add_relative_commission(meetings, data_path):
    # Add Department to cabinet
    meetings.loc[meetings['Name of DG'].isna(), 'Department'] = meetings['Name of cabinet'].str.extract( r'Cabinet of (Commissioner|Vice-President|High Representative|Executive Vice-President|President) ([^,]+)', expand=False)[1]

    # Add Department to DG

    data = pd.read_csv(data_path + 'DGs_relative_com.csv')
    com_relative_DG = dict(zip(data['Name of DG'], data['Commissioner']))
    meetings.loc[meetings['Name of cabinet'].isna(), 'Department'] = meetings['Name of DG'].map(com_relative_DG)
    return meetings


def merge_duplicated_meetings(meetings):

    # Define aggregate_tuple function
    def aggregate_tuple(x):
        return tuple( y  for sublist in x.dropna() for y in sublist)
    def aggregate_str(x):
        return tuple( y for y in x.dropna())

    # Use aggregate_tuple function in .agg()
    meetings = meetings.groupby(['Date', 'TR ID', 'Subject']).agg({
        'Name of cabinet': aggregate_str,
        'EC member': aggregate_tuple,
        'Title of EC representative': aggregate_tuple,
        'Name of DG': aggregate_str,
        'Department': aggregate_str,
        'Location' : 'first',
        'Name of interest representative' : 'first'
    }, default=pd.Series.mode).reset_index()

    meetings['Department'] = meetings['Department'].apply( lambda x : tuple(set(x)))

    return meetings

def preprocess_df(df, column):
    df[column] = df[column].str.strip()
    df[column] = df[column].str.lower()
    df[column] = df[column].apply(unidecode)
    df[column] = df[column].str.replace('&', 'and')
    df[column] = df[column].str.replace(';', ',')
    return df

def collapse_entities(meetings: pd.DataFrame, mapper: dict) -> pd.DataFrame:
    """Apply the mapper to collapse TR-ID tuples in `meetings` and return a new meetings DataFrame."""
    meetings = meetings.copy()
    meetings['TR ID'] = meetings['TR ID'].apply(
        lambda tpl: tuple(mapper.get(x, x) for x in tpl)
    )
    return meetings