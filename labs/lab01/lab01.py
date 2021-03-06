
import os

import pandas as pd
import numpy as np
import math

# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.

    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# Question # 1 
# ---------------------------------------------------------------------

def median(nums):
    """
    median takes a non-empty list of numbers,
    returning the median element of the list.
    If the list has even length, it should return
    the mean of the two elements in the middle.

    :param nums: a non-empty list of numbers.
    :returns: the median of the list.
    
    :Example:
    >>> median([6, 5, 4, 3, 2]) == 4
    True
    >>> median([50, 20, 15, 40]) == 30
    True
    >>> median([1, 2, 3, 4]) == 2.5
    True
    """
    
    length = int(len(nums))
    if (length%2==0) :
        return (nums[int(length/2)]+nums[int(length/2)-1])/2
    else :
        return nums[int(length/2)]
    #return ...


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance
    as integers is also i.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    elements as described above.

    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for i in range(len(ints) - 1):
        for j in np.arange(i,len(ints) - 1):
            diff = abs(ints[i] - ints[j])
            dis = abs(i-j)
            if diff == dis:
                return True

    return False


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def prefixes(s):
    """
    prefixes returns a string of every 
    consecutive prefix of the input string.

    :param s: a string.
    :returns: a string of every consecutive prefix of s.

    :Example:
    >>> prefixes('Data!')
    'DDaDatDataData!'
    >>> prefixes('Marina')
    'MMaMarMariMarinMarina'
    >>> prefixes('aaron')
    'aaaaaraaroaaron'
    """
    
    ss = s
    length = len(s)
    for i in np.arange(length - 1,0,-1):
        temp = s[:i] + ss
        ss = temp
    return ss
        


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def evens_reversed(N):
    """
    evens_reversed returns a string containing 
    all even integers from  1  to  N  (inclusive)
    in reversed order, separated by spaces. 
    Each integer is zero padded.

    :param N: a non-negative integer.
    :returns: a string containing all even integers 
    from 1 to N reversed, formatted as decsribed above.

    :Example:
    >>> evens_reversed(7)
    '6 4 2'
    >>> evens_reversed(10)
    '10 08 06 04 02'
    """
    ori_N = N
    if (N%2!=0) :
        N = N-1
    to_return = ""
    to_return += str(N)
    while N>2 :
        N -= 2
        to_return += " "
        if (ori_N > 9) :
            to_return += "0"
        to_return += str(N)
    return to_return


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of the line.

    :param fh: a file object to read from.
    :returns: a string of last characters from fh

    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """

    string = ""
    while True:
        line = fh.readline()
        string += line[-2:-1]
        if line == '':
            break
    return string


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """

    B = A.copy()
    length = len(B)
    for i in np.arange(length):
        B[i] += i**2
    return B


def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is divisble by 16.

    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.

    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 33]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """

    length = len(A)
    B = []
    for i in np.arange(length):
        if A[i]%16==0:
            B.append(True)
        else:
            B.append(False)
    return np.array(B)


def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """

    length = len(A)
    B = []
    for i in np.arange(length-1):
        B.append(round(((A[i+1]/A[i])-1),2))
    return np.array(B)


def arr_4(A):
    """
    Create a function arr_4 that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: an integer of the total number of shares.

    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """

    money = 20
    length = len(A)
    out = -1
    for i in np.arange(length):
        ori = math.floor(20/A[i])
        new = math.floor(money/A[i])
        if new>ori:
            out = i
            return out
        left = money%A[i]
        money += left
    return out


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def movie_stats(movies):
    """
    movies_stats returns a series as specified in the notebook.

    :param movies: a dataframe of summaries of
    movies per year as found in `movies_by_year.csv`
    :return: a series with index specified in the notebook.

    :Example:
    >>> movie_fp = os.path.join('data', 'movies_by_year.csv')
    >>> movies = pd.read_csv(movie_fp)
    >>> out = movie_stats(movies)
    >>> isinstance(out, pd.Series)
    True
    >>> 'num_years' in out.index
    True
    >>> isinstance(out.loc['second_lowest'], str)
    True
    """

    try:
        num_years = movies['Year'].nunique()
        tot_movies = movies['Number of Movies'].sum()
        yr_fewest_movies = movies.loc[movies['Number of Movies'] == movies['Number of Movies'].min()]['Year'].values[0]
        avg_gross = movies['Total Gross'].mean()
        avg = movies['Total Gross']/movies['Number of Movies']
        highest_per_movie = avg.max()
        second_lowest = movies.nsmallest(2, 'Total Gross')['#1 Movie'].values[0]
        hp = movies.loc[movies['#1 Movie'].str.contains('Harry Potter')]['Year']
        total = 0
        for i in np.arange(len(hp)):
            year = hp.values[i]+1
            total += movies.loc[movies['Year'] == year]['Number of Movies'].values[0]
        avg_after_harry = total/len(hp)
    except ValueError:
        print("Error")
        
    dict = {'num_years' : num_years, 
            'tot_movies' : tot_movies, 
            'yr_fewest_movies' : yr_fewest_movies,
            'avg_gross' : avg_gross,
            'highest_per_movie' : highest_per_movie,
            'second_lowest' : second_lowest,
            'avg_after_harry' : avg_after_harry} 
    return pd.Series(dict)
    

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a 
    properly formatted dataframe (as described in 
    the question).

    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data, 
    as specificed in the question statement.

    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """

    mf = open(fp)
    col = mf.readline()[:-1].split(",")
    fn = []
    ln = []
    he = []
    we = []
    lo = []
    while True:
        line = mf.readline()
        if line == '':
            break
        arr = line[:-1].split(",")
        if len(arr)>6:
            arr.remove('')
        #print(arr)
        fn.append(arr[0])
        ln.append(arr[1])
        we.append(float(arr[2].replace('"','')))
        he.append(float(arr[3].replace('"','')))
        lo.append((arr[4].replace('"',''))+","+(arr[5].replace('"','')))
    afn = np.array(fn)
    aln = np.array(ln)
    ahe = np.array(he)
    awe = np.array(we)
    alo = np.array(lo)
    #df = pd.DataFrame(np.array([afn,aln,ahe,awe,alo]),columns=col)
    d = {'first': afn, 'last': aln, 'weight': awe, 'height': ahe, 'geo': alo}
    df = pd.DataFrame(data=d)
    return df


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median'],
    'q02': ['same_diff_ints'],
    'q03': ['prefixes'],
    'q04': ['evens_reversed'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['movie_stats'],
    'q08': ['parse_malformed']
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
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
