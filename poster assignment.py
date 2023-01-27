# =============================================================================
#                               Importing Modules
# =============================================================================

import numpy as np
import errors as err
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt

import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



def read_data(first_path:str,second_path,name)->tuple:
    """
    This function reads in a file in the World Bank format and returns both the original and transposed format.
    The first element of the tuple is the original dataframe and the second element is the transposed dataframe.
    :param file_path: The file path to the World Bank data file.
    :return: A tuple containing the original dataframe and the transposed dataframe.
    """
    
    # read in the data file
    df_gdp = pd.read_csv(first_path,skiprows=4)
    df_unemp = pd.read_csv(second_path,skiprows=4)
    data_unemp=df_unemp[df_unemp["Country Name"]==name]
    data=df_gdp[df_gdp["Country Name"]==name]
    data=data.drop(["Country Name","Country Code","Indicator Name","Indicator Code","Unnamed: 66"],axis=1)
    data_unemp=data_unemp.drop(["Country Name","Country Code","Indicator Name","Indicator Code","Unnamed: 66"],axis=1)
    data=data.T.dropna()
    data_unemp=data_unemp.T.dropna()

    dataframe=pd.DataFrame()
    dataframe["GDP"]=data
    dataframe["unemployment"]=data_unemp
    dataframe=dataframe.reset_index()
    dataframe=dataframe.rename(columns={"index": "Year"})
   

    data_unemp_tranpose=data_unemp.T
    data_gdp_transpose=data.T
    return dataframe, data_gdp_transpose, data_unemp_tranpose
   

data,gdp_t,unemp_t = read_data("data.csv","unemp.csv","India")
data_1,gdp_1_t,unemp_1_t = read_data("data.csv","unemp.csv","United States")
data_2,gdp_2_t,unemp_2_t = read_data("data.csv","unemp.csv","Zimbabwe")


data = data.dropna()
data_1 = data_1.dropna()
data_2 = data_2.dropna()

print(data, data_1, data_2)



def poly_function(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def exp_growth(t, scale, growth):
    """
    Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1950))
    return f
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f
def logistics(t, scale, growth, t0):
    """ Computes logistics function with scale, growth rate
    and time of the turning point as free parameters
    """
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def poly(x, a, b, c, d):
    """Cubic polynominal for the fitting"""
    y = a*x**3 + b*x**2 + c*x + d
    return y



def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    # list to hold upper and lower limits for parameters
    uplow = []   
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper 


def curve_fitting(title_name, data):
    
    """
    This function takes in two arguments:
    title_name: a string that is used as the title for the plot.
    data: a pandas dataframe that contains the data to be plotted.
    It uses the curve_fit function from the scipy.optimize library to fit
    an exponential growth function to the data in the "unemployment" column
    of the dataframe. It then uses this function to forecast unemployment
    rates for the years 2030, 2040, and 2050 and plots the original data
    along with the forecasted values. It also prints the forecasted values
    and the upper and lower bounds of the uncertainty for each forecast.
    """
    
    popt, covar = opt.curve_fit(exp_growth, (data["Year"].values), data["unemployment"].values, p0= [4e8, 0.02])
    sigma = np.sqrt(np.diag(covar))
    year = np.arange(1991, 2031)
    print(year)
    forecast = exp_growth(year, *popt)
    print (popt, covar)
    print (data["GDP"])
    print(data["Year"])

    plt.figure()
    plt.plot(pd.to_numeric(data["Year"]), data["unemployment"].values, label="GDP")
    plt.plot(year, forecast, label="forecast")
    plt.title(title_name)
    plt.xlabel("Year")
    plt.ylabel("Unemployment")
    plt.legend()
    plt.show()
       
   #ploting data fro next decades
    print("Forcasted population")
       
    low, up = err.err_ranges(2030, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
       
    print("2030:", mean, "+/-", pm)
    print("2030 between ", low, "and", up)
       
    low, up = err.err_ranges(2040, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0

    print("2040:", mean, "+/-", pm)   
    print("2040 between ", low, "and", up)
       
    low, up = err.err_ranges(2050, exp_growth, popt, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
       
    print("2050:", mean, "+/-", pm)
    print("2050 between ", low, "and", up)
        
        
curve_fitting("India Forecast", data)
curve_fitting("United States Forecast", data_1)
curve_fitting("Zimbabwe Forecast", data_2)


def kmeans_cluster_plot(title_country, data, gdp_col, unemp_col):
    """
    This function takes in three parameters:
    title_country (str): The title of the plot, usually the name of the country.
    data (pandas DataFrame): The data that contains the columns for GDP and unemployment rate.
    gdp_col (str): The name of the column in the data that represents GDP.
    unemp_col (str): The name of the column in the data that represents unemployment rate.

    It creates a k-means clustering plot with 3 clusters of the normalized GDP and unemployment rate data,
    with the cluster centers marked and the silhouette score of the clustering printed.
    """
    
    df_ex = data[[gdp_col, unemp_col]].copy()
    max_val = df_ex.max()
    min_val = df_ex.min()
    df_ex = (df_ex - min_val) / (max_val - min_val)
    ncluster = 3
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_ex)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    print(cen)
    print(skmet.silhouette_score(df_ex, labels))
    plt.figure(figsize=(10.0, 10.0))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for l in range(ncluster):
        plt.plot(df_ex[labels==l][gdp_col], df_ex[labels==l][unemp_col], \
                 "o", markersize=3, color=col[l]), 
    for ic in range(ncluster):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10)
    plt.title(title_country)
    plt.xlabel(gdp_col)
    plt.ylabel(unemp_col)
    plt.show()
    

kmeans_cluster_plot("India Clustering", data, "GDP", "unemployment")
kmeans_cluster_plot("United States Clustering", data_1, "GDP", "unemployment")
kmeans_cluster_plot("Zimbabwe Clustering", data_2, "GDP", "unemployment")







