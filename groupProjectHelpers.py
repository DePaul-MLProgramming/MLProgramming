import warnings
import datetime
import pandas as pd
import numpy as np
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf
from ta.utils import dropna
from ta import add_all_ta_features
from sklearn import preprocessing, svm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, VarianceThreshold, RFE, RFECV
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb 
from sklearn.svm import SVC
from sklearn.datasets import make_classification, load_digits


# Use for validation with time series splits
n_splits = 10
tscv = TimeSeriesSplit(n_splits = n_splits)
def cross_val_classifier(model, X, y, params, n_splits, scoring, cv = tscv):
    """Performs cross validation and returns the best paramaeters and best score"""
    
    start_time = time.time()
    # Create the grid search
    grid_search = RandomizedSearchCV(estimator=model,
                                    param_distributions=params, 
                                    cv = cv, 
                                    n_iter=40,
                                    scoring = scoring, 
                                    verbose = 1)
    # Fit the model
    grid_search.fit(X, y)
    # Best_score
    best_score = grid_search.best_score_
    # Best parameters
    best_parameters = grid_search.best_params_
    
    end_time = time.time()
    run_time = (end_time-start_time)/60
    print(f"The total run time: {run_time} mins")
    print(f"The best parameters are: {best_parameters}")
    print(f"The best scores is: {best_score}")
    print("\n")
    print("\n")
    
    return

# Used for creating a confusion matrix and classification report
def cm_matrix_score(model, X_train, y_train, X_val, y_val):
    """Returns the confusion matrix and classification report"""
    
    start_time = time.time()
    
    # Fit the model
    model = model.fit(X_train, y_train)
    # Predict the accuracy 
    pred = model.predict(X_val)
    # Create the confusion matrix
    cm = confusion_matrix(y_val, pred, labels = model.classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= model.classes_)
    disp.plot()
    plt.show() 
    
    end_time = time.time()
    run_time = (end_time - start_time)/60
    # Create the classification matrix
    print(type(pred))
    print(metrics.classification_report(y_val, pred))
    print(f"The total run time: {run_time} mins")
    
    return

# Used for plotting feature importance.  Didn't end up using for project.  
def plot_feature_importance(model, training_data, n_features):
    """Plots the number of important features in a Random Forest Model"""
    
    feature_weights = model.feature_importances_
    feature_names = training_data.columns
    features_df = list(zip(feature_names, feature_weights))
    # Create the dataframe
    features_df1 = pd.DataFrame(features_df, columns=['Features', 'Weights'])
    # Sort Features by absolute value of the weights
    features_df2 = features_df1.sort_values(by = "Weights", key = pd.Series.abs, ascending = False)
    # Obtain the features and weights from data frame that will be used for plotting.
    features_plt = features_df2['Features']
    weights_plt = features_df2['Weights']
    
    
    # Plot the 20 most influential Weights. 
    fig, ax = plt.subplots(figsize=(16,9))
    #Horizontal bar plot
    ax.barh(features_plt[:n_features], weights_plt[:n_features])
    plt.title("20 Most Important Features")
    plt.show()
    
    return features_df2

# Calculates the silhouette score for clustering.  Didn't use for project.  
def calc_silhouette(clusters, km, transformed_values):
    """Calculate the silhouette score"""
    
    # Fit the model
    km.fit_predict(transformed_values)
    # Calculate the Silhouette score
    score = silhouette_score(transformed_values, km.labels_, metric = 'euclidean')
    print(f"The silhouette score for {clusters} clusters is: {score}.")
    
    return

# Performs clustering analysis.  Didn't use for project.  
# Reference: https://towardsdatascience.com/rfmt-segmentation-using-k-means-clustering-76bc5040ead5
def clustering(n_clusters, transformed_X_values, columns):
    """Performs K means clustering"""
    time_start = time.time()
    
    # Create dataframe
    X_clust = pd.DataFrame(transformed_X_values, columns=columns)
    km = KMeans(n_clusters = n_clusters, 
               random_state =123, 
               n_init = 500, 
               max_iter = 300)
    km.fit(transformed_X_values)
    # Assign cluster labels
    cluster_labels = km.labels_  
    # Assign cluster labels to the original pre_transformed data set
    data_clusters = X_clust.assign(Cluster = cluster_labels) 
    # Group Dataset by k-means cluster
    data_clusters = data_clusters.groupby(['Cluster']).agg('mean')
    # Determine the cluster size
    cluster_size = X_clust.assign(Cluster = cluster_labels).groupby(cluster_labels)['Target'].count() 
    # Table of clusters
    cluster_table = data_clusters.iloc[:,:21]
    
    print(f"The cluster sizes are:") 
    print(cluster_size)
    print("\n")
    print(f"The clusters are:")
    print(cluster_table)
    print("\n")
        
    # Cluster heatmap
    # Initialize a plot with a figure size of 8 by 2 inches 
    plt.figure(figsize=(20, 10))
    # Add the plot title
    plt.title('Relative importance of attributes')
    # Plot the heatmap
    sns.heatmap(data=data_clusters.iloc[:,:59], fmt='.2f', cmap='RdYlGn', xticklabels=True)
    plt.show()

    time_end = time.time()
    runt_time = (time_end - time_start)/60
    print(f"Runtime is: {run_time} mins.")
          
    return km


# Reference: https://towardsdatascience.com/time-series-analysis-for-machine-learning-with-python-626bee0d0205
# Used to create plot of time series.  
def plot_ts(ts, plot_ma=True, plot_intervals=True, window=7,
            figsize=(15,5)):
    """Plots the moving average bollinger bands """
    rolling_mean = ts.rolling(window=window).mean()    
    rolling_std = ts.rolling(window=window).std()
    plt.figure(figsize=figsize)    
    #plt.title(ts.name)    
    plt.plot(ts[window:], label='Actual values', color="black")    
    if plot_ma:
        plt.plot(rolling_mean, 'g', label='MA'+str(window),
               color="red")    
    if plot_intervals:
        lower_bound = rolling_mean - (1.96 * rolling_std)
        lower_bound = np.array(lower_bound).flatten()
        upper_bound = rolling_mean + (1.96 * rolling_std)
        upper_bound = np.array(upper_bound).flatten()
        
    plt.fill_between(x=ts.index, y1=lower_bound, y2=upper_bound,
                    color='lightskyblue', alpha=0.4)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# Used to show outliers of time series plots
def find_outliers(ts, perc=0.01, figsize=(15,5)):
    ## fit svm
    scaler = preprocessing.StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
    model = svm.OneClassSVM(nu=perc, kernel="rbf", gamma=0.01)
    model.fit(ts_scaled)
    ## dtf output
    ts = ts.squeeze()
    dtf_outliers = ts.to_frame(name="ts")
    dtf_outliers["index"] = range(len(ts))
    dtf_outliers["outlier"] = model.predict(ts_scaled)
    dtf_outliers["outlier"] = dtf_outliers["outlier"].apply(lambda
                                              x: 1 if x==-1 else 0)
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title="Outliers detection: found"
           +str(sum(dtf_outliers["outlier"]==1)))
    ax.plot(dtf_outliers["index"], dtf_outliers["ts"],
            color="black")
    ax.scatter(x=dtf_outliers[dtf_outliers["outlier"]==1]["index"],
               y=dtf_outliers[dtf_outliers["outlier"]==1]['ts'],
               color='red')
    ax.grid(True)
    plt.show()
    return dtf_outliers


# Used to remove outliers
def remove_outliers(ts, outliers_idx, figsize=(15,5)):
    ts_clean = ts.copy()
    ts_clean.loc[outliers_idx] = np.nan
    ts_clean = ts_clean.interpolate(method="linear")
    ax = ts.plot(figsize=figsize, color="red", alpha=0.5,
         title="Remove outliers", label="original", legend=True)
    ts_clean.plot(ax=ax, grid=True, color="black",
                  label="interpolated", legend=True)
    plt.show()
    return ts_clean

