import os
import pandas as pd
import numpy as np


def create_transparency_register(path):
    file_names = os.listdir(os.path.join (path, 'Transparency_register' ))
    file_names.sort() # Sort to keep the latest information about organization

    TR = pd.DataFrame()

    mapper_previous2new_category = {'Other organisations' : 'Other organisations, public or mixed entities',
                                    'Regional structures': 'Associations and networks of public authorities',
                                    'Other public or mixed entities, created by law whose purpose is to act in the public interest' : 'Other organisations, public or mixed entities',
                                    'Other sub-national public authorities' : 'Associations and networks of public authorities',
                                    'Transnational associations and networks of public regional or other sub-national authorities' : 'Associations and networks of public authorities',
                                    'Self-employed consultants': 'Self-employed individuals' }

    mapper_previous2new_column = {
    'Registration date:':'Registration date',
    'Subsection':'Category of registration',
    '(Organisation) name':'Name',
    'Legal status:':'Form of the entity',
    'Website address:':'Website URL',
    'Belgium office address':'EU office address',
    'Belgium office city':'EU office city',
    'Belgium office post code': 'EU office post code',
    'Belgium office post box':'EU office post box',
    'Belgium office phone':'EU office phone',
    'Goals / remit':'Goals',
    'EU initiatives':'EU legislative proposals/policies',
    'Relevant communication':'Communication activities',
    'High-level groups':'Expert Groups',
    'Inter groups':'Intergroups and unofficial groupings',
    'Number of persons involved:':'Members',
    'Full time equivalent (FTE)':'Members FTE',
    'Number of EP accredited persons':'Number of EP acredited Person',

    'Membership':'Is member of: List of associations, (con)federations, networks or other bodies of which the organisation is a member',

    'Member organisations':'Organisation Members: List of organisations, networks and associations that are the members and/or  affiliated with the organisation'
    }

    mapper_shorten= {'Head office country' : 'TR Country',
                        'Name': 'TR Name'}



    for file in file_names:

        try :
            # New version of the transparency register
            df = pd.read_excel(os.path.join( path, 'Transparency_register', file), engine='xlrd', index_col = 'Identification code' )
        except :
            # Previous version of the transparency register
            df = pd.read_excel(os.path.join( path, 'Transparency_register', file), engine='xlrd', index_col = 'Identification number:' )
            df.index = df.index.rename('Identification code')
            df.rename(columns = mapper_previous2new_column, inplace = True)

        TR = pd.concat([TR, df]).groupby('Identification code').last()
    TR.rename(columns = mapper_shorten, inplace = True)

    TR['TR Country'] = TR['TR Country'].str.upper()
    TR['Level of interest'] = TR['Level of interest'].str.title()
    TR['Category of registration'] = TR['Category of registration'].apply( lambda  x : mapper_previous2new_category.get(x, x))
    return(TR)

def TransparencyRegister(path, columns = None,reload = False):
    if reload :
        TR = create_transparency_register(path)
        TR.to_csv(os.path.join( path, 'Transparency_register.csv'))
    else :
        TR = pd.read_csv(os.path.join( path, 'Transparency_register.csv'), index_col = 'Identification code')
    if columns is not None:
        TR = TR[columns]
    return(TR)
