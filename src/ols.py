import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from scipy.stats import yeojohnson, skew
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_skewed_columns(df):
    """
    :param df: dataframe where the skewed columns need to determined
    :return: skew_cols: dataframe with the skewed columns
    """
    skew_limit = 0.5  # define a limit above which we will log transform
    skew_vals = df.skew()
    # Showing the skewed columns
    skew_cols = (skew_vals
                 .sort_values(ascending=False)
                 .to_frame()
                 .rename(columns={0: 'Skew'})
                 .query('abs(Skew) > {}'.format(skew_limit)))
    return skew_cols

def get_similar_value_cols(df, percent=90):
    """
    :param df: input data in the form of a dataframe
    :param percent: integer value for the threshold for finding similar values in columns
    :return: sim_val_cols: list of columns where a singular value occurs more than the threshold
    """
    count = 0
    sim_val_cols = []
    for col in df.columns:
        percent_vals = (df[col].value_counts()/len(df)*100).values
        # filter columns where more than 90% values are same and leave out binary encoded columns
        if percent_vals[0] > percent :
            sim_val_cols.append(col)
            count += 1
    print("Total columns with majority singular value shares: ", count, sim_val_cols)
    
    return sim_val_cols


def one_hot_encod(df : pd.DataFrame , col : list , to_drop = None):
    """
    :param df: input data in the form of a dataframe
    :param columns: list of columns to encode
    :return: df: dataframe with the encoded columns
    """

    if to_drop == 'first':
        dummies  =  pd.get_dummies(df[col], drop_first=True, prefix_sep = '', prefix = '', dtype = int)
    elif to_drop is not None:
        dummies  =  pd.get_dummies(df[col],  prefix_sep = '', prefix = '', dtype = int)
        dummies.drop(columns = to_drop, inplace = True)

    else :
        dummies  =  pd.get_dummies(df[col], prefix_sep = '', prefix = '', dtype = int)

    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    return df 
    #, list(dummies.columns))



def remove_outliers(df, col :str, threshold):
    df_ = df[col].value_counts(normalize = True).copy()*100

    cat_to_remove = list( df_[df_<threshold].index)
    com_to_remove = list(df[df[col].isin(cat_to_remove)].index)
    print('removed category: ' , cat_to_remove)
    print('number of outliers: ', len(com_to_remove))

    return(df[~df[col].isin(cat_to_remove)])


def transform_skewed(df, columns : list, method : str = 'log'):
    for col in columns:
        if method == 'yeo':
            df[col] = yeojohnson(df[col])[0]
        elif method == 'log':

            #df = df[df[col] + 1>0]
            df[col] = df[col].apply(lambda x : np.log(x +1))
    return(df)


def xy_heatmap_corr (df , col1, col2, **kwargs ):
    import seaborn as sns
    try:
        plt.close()
    except:
        pass
    sns.heatmap(df[col1+col2].corr( ).loc[col1][col2] ,  cmap = 'inferno_r' , vmin =0, vmax= 1 , **kwargs)
    plt.tight_layout()
    plt.show()

def replace_by_GUO_data(x):
    if x['GUO Type'] in ['Corporate', 'Bank', 'Insurance company', 'Financial company']:

        if pd.notna(x['GUO Revenue']) or pd.notna(x['GUO Assets']) or pd.notna(x['GUO Nb employees']):
            return pd.Series({
                        'Revenue': 1e3 * x['GUO Revenue'],
                        'Assets': 1e3 * x['GUO Assets'],
                        'Nb employees': x['GUO Nb employees'],
                        'NACE': x['GUO NACE']
                    })
        else:
            return x[['Revenue', 'Assets', 'Nb employees', 'NACE']]

    else:
        return x[['Revenue', 'Assets', 'Nb employees', 'NACE']]

## Analysis of variance

def pr_F(current_model , variables, tagret, basetable):
    X = basetable[variables]
    y = basetable[target]
    model = sm.OLS(y, X.assign(cont = 1)).fit()
    if model.df_model > current_model.df_model:
        anova_results = sm.stats.anova_lm(current_model, model )
    else:
        anova_results = sm.stats.anova_lm( model , current_model)
    return(anova_results['Pr(>F)'][1])

def variables_to_add(current_variables, candidate_variables, target, basetable, threshold = 0.05):
    variables  = []

    X = basetable[current_variables]
    y = basetable[target]
    current_model = sm.OLS(y, X.assign(cont = 1)).fit()
    for v in candidate_variables:
        pr_F_v = pr_F( current_model , current_variables +[v], target, basetable)
        if pr_F_v < threshold:
            variables.append(v)
    return variables

def variables_to_remove(current_variables, target, basetable, threshold =0.05):

    variables =[]
    X = basetable[current_variables]
    y = basetable[target]
    current_model = sm.OLS(y, X.assign(cont = 1)).fit()
    for v in current_variables:
        red_variables = list( set(current_variables) - set([v]))
        pr_F_v = pr_F( current_model , red_variables, target, basetable)
        if pr_F_v > threshold or np.isnan(pr_F_v) :
            variables.append(v)
    return variables


def stepwise_selection(candidate_variables, target, basetable):
    current_variables = []
    nb_itt_max = len(candidate_variables)
    while True:
        c = 0
        next_var = variables_to_add(current_variables, candidate_variables, target, basetable, threshold = teta_in)
        print(next_var)
        if next_var ==[]:
            c+=1
        else :
            current_variables = current_variables + next_var
            for v in next_var:
                candidate_variables.remove(v)
        next_var  = variables_to_remove(current_variables, target, basetable, threshold = teta_out)
        print(next_var)
        if next_var == []:
            c+=1
        else:
            for v in next_var:
                current_variables.remove(v)
                candidate_variables.append(v)

        if c == 2:
            break
    return(current_variables)



