import numpy as np
import pandas as pd

def ninteraction(df, drop=False):
    """
    Compute a unique numeric id for each unique row in a data frame.
    """
    if len(df.columns) == 0:
        return np.zeros(len(df))
    if drop:
        df = df.drop_duplicates()
    return pd.factorize(df.apply(tuple, axis=1))[0]

def add_margins(data, vars, margins=True):
    """
    Add margins to a data frame.
    """
    if not margins or not vars:
        return data
    
    all_vars = []
    for v in vars:
        if isinstance(v, list):
            all_vars.extend(v)
        else:
            all_vars.append(v)
    
    margin_vars = [v for v in all_vars if v in data.columns]
    if not margin_vars:
        return data
    
    # Add a copy of the data with each variable set to None
    margin_dfs = [data]
    for v in margin_vars:
        df_copy = data.copy()
        df_copy[v] = None
        margin_dfs.append(df_copy)
    
    return pd.concat(margin_dfs, ignore_index=True)

def cross_join(df1, df2):
    """
    Compute a cross join between two data frames.
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1
    merged = pd.merge(df1, df2, on='_tmpkey')
    merged = merged.drop('_tmpkey', axis=1)
    return merged

def match(x, table, start=0):
    """
    Return a vector of the positions of (first) matches of x in table.
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(table, pd.Series):
        table = pd.Series(table)
    
    result = pd.Series(np.nan, index=x.index)
    for i, val in enumerate(table, start=start):
        mask = (x == val) & result.isna()
        result[mask] = i
    return result.astype(int)

def join_keys(x, y, by):
    """
    Join two data frames by common keys.
    """
    if not by:
        return {'x': pd.Series(), 'y': pd.Series()}
    
    x_vals = x[by].apply(tuple, axis=1)
    y_vals = y[by].apply(tuple, axis=1)
    
    return {'x': x_vals, 'y': y_vals} 