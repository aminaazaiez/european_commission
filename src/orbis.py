import os
import pandas as pd
import numpy as np

#from classes.european_commission import *


def search_in_batches(orbis_path, company_name):
    files  = os.listdir(orbis_path + 'Company_Matchs')
    for file_name in files :
        if 'organization_names' in file_name :
            df = pd.read_excel( orbis_path + '/Company_Matchs/'+file_name, engine = "openpyxl", usecols = ['Company name','Matched BvD ID','Matched company name'])
            if company_name in list(df['Company name']):
                print(company_name, 'in', file_name)


class Orbis:

    def load_matched_names (self,reload = False ):

        if reload :
            # Load company name batch with TR ID

            company_name_batchs = pd.DataFrame()
            files  = os.listdir(self.orbis_path  + 'Company_name_batchs')
            files.sort()
            for file_name in files :
                if 'company_names' in file_name :
                    df = pd.read_csv( self.orbis_path + '/Company_name_batchs/'+file_name,sep = '\t', usecols = ['TR ID', 'Name'])

                    company_name_batchs =  pd.concat([company_name_batchs , df ], ignore_index = True)

            #Load company name matchs with BvD ID

            matched_names = pd.DataFrame()
            files  = os.listdir(self.orbis_path  + 'Company_Matchs')
            files.sort()
            for file_name in files :
                if 'company_names' in file_name :
                    df = pd.read_excel( self.orbis_path  + '/Company_Matchs/'+file_name, engine = "openpyxl", usecols = ['Company name','Matched BvD ID'])
                    # # Check that names in df corresponds to names EU.entities (problem with double spaces and " )
                    # for name in list(df['Company name']):
                    #     if name not in list(EU.entities['Name']):
                    #         print(name , file_name)


                    matched_names =  pd.concat([matched_names , df ], ignore_index = True)
            matched_names.rename(columns = {'Matched BvD ID' : 'BvD ID'}, inplace = True)

            # Create a dataframe with TR ID and BvD ID
            self.matched_names  = pd.concat([company_name_batchs , matched_names], axis  =1)

            self.matched_names.to_csv(self.orbis_path  + 'Orbis_matchs.csv')

        else :
            self.matched_names = pd.read_csv(self.orbis_path  + 'Orbis_matchs.csv', index_col = 'TR ID')
            self.matched_names.drop(['Unnamed: 0'], axis = 1, inplace = True )





    def load_company_data (self, check = False):

        self.company_data = pd.read_excel( self.orbis_path  + 'Company_data/Orbis_company_data.xlsx', engine = "openpyxl", sheet_name = 'Results')

        self.company_data.drop(['Unnamed: 0'], axis = 1, inplace = True)


        self.company_data.rename(columns = {'Company name Latin alphabet' : 'Name',
                                        'Country' : 'Orbis Country',
                                        'Country ISO code' : 'Country ISO',
                                        'City\nLatin Alphabet' : 'City',
                                        'BvD ID number' : 'BvD ID',
                                        'BvD sectors' :'BvD sectors' ,
                                        'NAICS 2022, core code - description' : 'NAICS',
                                        'Last avail. year' : 'Last year',
                                        'Operating revenue (Turnover)\nth USD Last avaFreil. yr' : 'Revenue',
                                        'Total assets\nth USD Last avail. yr' : 'Assets',
                                        'Number of employees\nLast avail. yr' : 'Nb employees',
                                        'No of companies in corporate group' : 'corporate group size',
                                        'Entity type' : 'Entity type',
                                        'GUO - Name' : 'GUO Name',
                                        'GUO - BvD ID number' : 'GUO BvD ID number',
                                        'GUO - Legal Entity Identifier (LEI)' : 'GUO LEI',
                                        'GUO - Country ISO code' : 'GUO Country ISO',
                                        'GUO - City' : 'GUO City',
                                        'GUO - Type' : 'GUO Type',
                                        'GUO - NAICS, text description' : 'GUO NAICS',
                                        'GUO - Operating revenue (Turnover)\nm USD' : 'GUO Revenue',
                                        'GUO - Total assets\nm USD' : 'GUO Assets',
                                        'GUO - Number of employees' : 'GUO Nb employees',
                                        'GUO - NACE,Core code': 'GUO NACE',
                                        'NACE Rev. 2 main section' : 'NACE',
                                        'NACE Rev. 2, core code (4 digits)': 'NACE core'} , inplace = True)
        self.company_data.replace('n.a.', np.nan, inplace = True)
        self.company_data.replace('-', np.nan , inplace = True)

        for column in self.company_data:
            if self.company_data[column].dtype != float :
                self.company_data[column] = self.company_data[column].str.replace('&', 'and')
                self.company_data[column] = self.company_data[column].str.replace(';', ',')



        NACE_structure = {'A - Agriculture, forestry and fishing' : [100,399],
            'B - Mining and quarrying' : [500,999],
            'C - Manufacturing' : [1000,3399] ,
            'D - Electricity, gas, steam and air conditioning supply' : [3500, 3599],
            'E - Water supply, sewerage, waste management and remediation activities' : [3600,3999],
            'F - Construction' : [ 4100,4399],
            'G - Wholesale and retail trade, repair of motor vehicles and motorcycles' : [4500,4799],
            'H - Transportation and storage' : [4900,5399] ,
            'I - Accommodation and food service activities' : [5500,5699],
            'J - Information and communication' : [5800,6399] ,
            'K - Financial and insurance activities' : [ 6400,6699],
            'L - Real estate activities' : [ 6800,6899],
            'M - Professional, scientific and technical activities' : [6900,7599],
            'N - Administrative and support service activities' : [7700,8299] ,
            'O - Public administration and defence, compulsory social security' :[8400,8499],
            'P - Education' : [8500,8599],
            'Q - Human health and social work activities' : [8600,8899],
            'R - Arts, entertainment and recreation' : [9000,9399],
            'S - Other service activities': [9400,9699]}

        def replace_GUO_NACE(x):
            if pd.notna(x):
                x = int(x)
                for main_section , range_code in NACE_structure.items():
                    if range_code[0] <= x <=range_code[1]:
                        return(main_section)
                print(f'Problem with NACE code {range_code[0] } <= {x} <={range_code[1]} ,  {range_code[0] <= x <=range_code[1]}')

        self.company_data['GUO NACE'] =self.company_data['GUO NACE'].apply(replace_GUO_NACE)



        if check :
            # Check comapnies with unknown BvD
            for name, ID in zip (self.company_data['Name'] , self.company_data['BvD ID']):
                if len(self.company_data[self.company_data['Name'] == name].dropna(subset = 'BvD ID')) == 0:
                    print(name , 'in company data has no BvD ID')
                else :
                    self.company_data.dropna(subset = 'BvD ID', inplace = True)


            if len(self.company_data[self.company_data['BvD ID'].str.contains('&')]) > 1 :
                print('There is BvD ID number containing a "&"')



        self.company_data.drop_duplicates(subset = 'BvD ID', inplace = True)
        self.company_data.dropna(subset = 'BvD ID', inplace = True)





    def __init__(
        self,
        orbis_path : str,
        matched_names : pd.DataFrame = pd.DataFrame(),
        company_data : pd.DataFrame = pd.DataFrame(),
        check = False,
        reload = False,
        ):
        self.orbis_path = orbis_path
        self.matched_names = matched_names
        self.company_data = company_data


        self.load_company_data( check = check)
        self.load_matched_names(reload = reload)


        if check :
            # Check that all BvD in orbis data are in matchs
            for orga, ID in zip (self.company_data['Name'], self.company_data['BvD ID']):
                if ID not in set(self.matched_names['BvD ID']):
                    print( 'Entity %s with ID %s is in orbis company data but not in orbis matchs' %(orga , ID))

            # Check that all BvD in matchs are in Orbis data
            IDS =  set(self.matched_names.dropna(subset = 'BvD ID')['BvD ID']) - set(self.company_data['BvD ID'].dropna())
            if len(IDS) >0:
                print('BvD in matchs are not in Orbis data \n IDS' )
            print(self.matched_names[self.matched_names['BvD ID'].isin(IDS)][['Company name', 'BvD ID']])



    def get_BvD_matched_companies(self):
        IDS = ""
        for ID in self.matched_names.dropna(subset = 'BvD ID')['BvD ID']:
            IDS += "%s;"%ID
        IDS =IDS[:-1]
        return(IDS)
#
# orbis = Orbis()
# IDS = orbis.get_BvD_matched_companies()
# #open text file
# text_file = open( orbis_path + 'BvD.txt', "w")
#
# #write string to file
# text_file.write(IDS)
#
# #close file
# text_file.close()
