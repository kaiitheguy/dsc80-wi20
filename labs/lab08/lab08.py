import pandas as pd
import numpy as np
import seaborn as sns
import os
from scipy import stats

from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def best_transformation():
    """
    Returns an integer corresponding to the correct option.

    :Example:
    >>> best_transformation() in [1,2,3,4]
    True
    """

    # take log and square root of the dataset
    # look at the fit of the regression line (and R^2)
    
    homeruns_fp = os.path.join('data', 'homeruns.csv')
    homeruns = pd.read_csv(homeruns_fp)
    x = homeruns['Year']
    y0 = homeruns['Homeruns']
    y1 = y0**(.5)
    y2 = np.log(y0)
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x, y1)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x, y2)
    print(slope1)
    print(slope2)
    if slope1>3 and slope3>3:
        return 4
    elif abs(slope1-slope2)<1:
        return 3
    elif slope1>slope2:
        return 1
    else:
        return 2

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------


def create_ordinal(df):
    """
    create_ordinal takes in diamonds and returns a dataframe of ordinal
    features with names ordinal_<col> where <col> is the original
    categorical column name.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_ordinal(diamonds)
    >>> set(out.columns) == {'ordinal_cut', 'ordinal_clarity', 'ordinal_color'}
    True
    >>> np.unique(out['ordinal_cut']).tolist() == [0, 1, 2, 3, 4]
    True
    """
    
    #enc = OrdinalEncoder()
    #enc.fit([diamonds['cut'],diamonds['clarity'],diamonds['color']])
    #enc.transform([diamonds['cut'],diamonds['clarity'],diamonds['color']])
    df = df.select_dtypes(exclude=['int64','float','int'])
    reldf = df.apply(ord_col)
    reldf = reldf.rename(columns=lambda s: 'ordinal_'+s)
    return reldf

def ord_col(col):
    rel = col.copy()
    uni = col.unique()
    for i in np.arange(len(rel)):
        idx = np.where(uni==rel[i])[0][0]
        rel[i] = idx
    return rel
# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def create_one_hot(df):
    """
    create_one_hot takes in diamonds and returns a dataframe of one-hot 
    encoded features with names one_hot_<col>_<val> where <col> is the 
    original categorical column name, and <val> is the value found in 
    the categorical column <col>.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_one_hot(diamonds)
    >>> out.shape == (53940, 20)
    True
    >>> out.columns.str.startswith('one_hot').all()
    True
    >>> out.isin([0,1]).all().all()
    True
    """
    
    df = df.select_dtypes(exclude=['int64','float','int'])
    cols = df.columns
    for i in np.arange(len(cols)):
        #print(df['color'])
        temp_df = hot_col(df[cols[i]])
        df = pd.concat([df,temp_df], axis=1, sort=False)
    #reldf = df.apply(ord_col)
    #reldf = reldf.rename(columns=lambda s: 'ordinal_'+s)
    df = df.drop(cols,axis=1)
    return df

def hot_col(col):
    uni = col.unique()
    lists = []
    for j in np.arange(len(uni)):
        strj = uni[j]
        rel = col.copy()
        for i in np.arange(len(rel)):
            if (strj==rel[i]):
                idx = 1
            else:
                idx = 0
            rel[i] = idx
        lists.append(rel)
    arrs = np.array(lists)
    #print(uni)
    #print(len(arrs[0]))
    #df = pd.DataFrame.from_records(arrs)
    df_T = pd.DataFrame(arrs)
    df = df_T.T
    df.columns = uni
    df = df.rename(columns=lambda s: 'one_hot_'+col.name+'_'+s)
    return df

def create_proportions(df):
    """
    create_proportions takes in diamonds and returns a 
    dataframe of proportion-encoded features with names 
    proportion_<col> where <col> is the original 
    categorical column name.

    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_proportions(diamonds)
    >>> out.shape[1] == 3
    True
    >>> out.columns.str.startswith('proportion_').all()
    True
    >>> ((out >= 0) & (out <= 1)).all().all()
    True
    """

    df = df.select_dtypes(exclude=['int64','float','int'])
    reldf = df.apply(pro_col)
    reldf = reldf.rename(columns=lambda s: 'proportion_'+s)
    return reldf

def pro_col(col):
    rel = col.copy()
    val = col.value_counts()
    tol = len(rel)
    
    for i in np.arange(len(rel)):
        num = val[rel[i]]
        rel[i] = num/tol
    return rel

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    """
    create_quadratics that takes in diamonds and returns a dataframe 
    of quadratic-encoded features <col1> * <col2> where <col1> and <col2> 
    are the original quantitative columns 
    (col1 and col2 should be distinct columns).

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_quadratics(diamonds)
    >>> out.columns.str.contains(' * ').all()
    True
    >>> ('x * z' in out.columns) or ('z * x' in out.columns)
    True
    >>> out.shape[1] == 15
    True
    """
        
    df = df.select_dtypes(include=['float'])
    cols = df.columns
    for i in np.arange(len(cols)):
        col_i = df[cols[i]]
        for j in np.arange(i+1,len(cols),1):
            col_j = df[cols[j]]
            df.insert(0,cols[i]+' * '+cols[j],col_i*col_j,True)       
    df = df.drop(cols,axis=1)
    return df


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def comparing_performance():
    """
    Hard coded answers to comparing_performance.

    :Example:
    >>> out = comparing_performance()
    >>> len(out) == 6
    True
    >>> import numbers
    >>> isinstance(out[0], numbers.Real)
    True
    >>> all(isinstance(x, str) for x in out[2:-1])
    True
    >>> 0 <= out[-1] <= 1
    True
    """

    # create a model per variable => (variable, R^2, RMSE) table

    return [0.9697467062649333, 0.08244625824198684, 'price',
            'carat*price', 'quadratic features', 0.11052802310683463]

# ---------------------------------------------------------------------
# Question # 6, 7, 8
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    def transformCarat(self, data):
        """
        transformCarat takes in a dataframe like diamonds 
        and returns a binarized carat column (an np.ndarray).

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transformCarat(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> transformed[172, 0] == 1
        True
        >>> transformed[0, 0] == 0
        True
        """

        return ...
    
    def transform_to_quantile(self, data):
        """
        transform_to_quantiles takes in a dataframe like diamonds 
        and returns an np.ndarray of quantiles of the weight 
        (i.e. carats) of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds.head(10))
        >>> transformed = out.transform_to_quantile(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> 0.2 <= transformed[0,0] <= 0.5
        True
        >>> np.isclose(transformed[1,0], 0, atol=1e-06)
        True
        """

        return ...
    
    def transform_to_depth_pct(self, data):
        """
        transform_to_volume takes in a dataframe like diamonds 
        and returns an np.ndarray consisting of the approximate 
        depth percentage of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds').drop(columns='depth')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_to_depth_pct(diamonds)
        >>> len(transformed.shape) == 1
        True
        >>> np.isclose(transformed[0], 61.286, atol=0.0001)
        True
        """

        return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['best_transformation'],
    'q02': ['create_ordinal'],
    'q03': ['create_one_hot', 'create_proportions'],
    'q04': ['create_quadratics'],
    'q05': ['comparing_performance'],
    'q06,7,8': ['TransformDiamonds']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
