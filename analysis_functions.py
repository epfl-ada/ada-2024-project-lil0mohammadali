import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd

def get_general_ch_statistics(filtered_df, cols_to_keep = ['dislike_count','duration','like_count','view_count','num_comms'], channel = False):

    """
    Group data by channel id and compute some general statistics.
    
    --------------------------------------------------------
    Parameters:
    filtered_df (pl.Dataframe):
        dataframe with the data over wich we want to compute the statistics.
        
    cols_to_keep (list of strings):
        column indices over which we want to compute the statistics.
        
    channel (bool):
        Boolean that indicates whether the channel id is contained in a column called "channel" or "channel_id".       
    --------------------------------------------------------
    Returns (tuple):
        A tuple containing :
        counts (pl.Dataframe):
            Array with the number of entries for each channel.
        
        means (pl.Dataframe):
            Array with the mean of each metric for each channel.
        
        stdevs (pl.Dataframe):
            Array with the standard deviation of each metric for each channel.
            
        medians (pl.Dataframe):
            Array with the median of each metric for each channel.
    
    """

    if channel :
        col_to_group_by = 'channel'
    else : 
        col_to_group_by = 'channel_id'

    counts = filtered_df.group_by(col_to_group_by, maintain_order=True).len()
    means = filtered_df.group_by(col_to_group_by, maintain_order=True).agg(pl.col(cols_to_keep).mean())
    stdevs = filtered_df.group_by(col_to_group_by, maintain_order=True).agg(pl.col(cols_to_keep).std())
    medians = filtered_df.group_by(col_to_group_by, maintain_order=True).agg(pl.col(cols_to_keep).median())
    counts = counts.rename({'len':'counts'})
    
    return counts,means,stdevs,medians

def ttest_between_two_channels (df, channel_id_1, channel_id_2, variable, channel = False):
    
    """
    Perform a t-test between two different channels to asses the independence of their means.
    
    --------------------------------------------------------
    Parameters:
    df (pl.Dataframe):
        dataframe with the data grouped by chanel over wich we want to compute the t test
        
    channel_id_1 (string):
        id of the first channel we want to evaluate
            
    channel_id_2 (string):
        id of the second channel we want to evaluate 
                
    channel (bool):
        Boolean that indicates whether the channel id is contained in a column called "channel" or "channel_id"       
    --------------------------------------------------------
    Returns (tuple):
        A tuple containing the t-statistic and the p-value of the test.
    """
    
    
    if channel :
        col_with_channel_id = 'channel'
    else : 
        col_with_channel_id = 'channel_id'
        
    if (len((df[col_with_channel_id] == channel_id_1).unique()) == 1 or len((df[col_with_channel_id] == channel_id_2).unique()) == 1):
        return "Invalid channel id"
    
    ch_1 = df.filter(pl.col(col_with_channel_id) == channel_id_1)[variable]
    ch_2 = df.filter(pl.col(col_with_channel_id) == channel_id_2)[variable]
    ch_1 = ch_1.drop_nulls()
    ch_2 = ch_2.drop_nulls()
    ch_1 = ch_1.drop_nans()
    ch_2 = ch_2.drop_nans()
    return stats.ttest_ind(ch_1, ch_2, equal_var=False)

def Ftest_between_two_channels (df, channel_id_1, channel_id_2, variable, channel = False):
    
    """
    Perform an F-test between two different channels to asses the independence of their variances.
    
    --------------------------------------------------------
    Parameters:
    df (pl.Dataframe):
        dataframe with the data grouped by chanel over wich we want to compute the t test
        
    channel_id_1 (string):
        id of the first channel we want to evaluate
            
    channel_id_2 (string):
        id of the second channel we want to evaluate 
                
    channel (cool):
        Boolean that indicates whether the channel id is contained in a column called "channel" or "channel_id"       
    --------------------------------------------------------
    Returns (tuple):
        A tuple containing the F-statistic and the p-value of the test.
    """
    
    
    if channel :
        col_with_channel_id = 'channel'
    else : 
        col_with_channel_id = 'channel_id'
        
    if (len((df[col_with_channel_id] == channel_id_1).unique()) == 1 or len((df[col_with_channel_id] == channel_id_2).unique()) == 1):
        return "Invalid channel id"
    
    ch_1 = df.filter(pl.col(col_with_channel_id) == channel_id_1)[variable]
    ch_2 = df.filter(pl.col(col_with_channel_id) == channel_id_2)[variable]
    ch_1 = ch_1.drop_nulls()
    ch_2 = ch_2.drop_nulls()
    ch_1 = ch_1.drop_nans()
    ch_2 = ch_2.drop_nans()
    return stats.f_oneway(ch_1, ch_2)

def plot_video_characteristics_for_given_channel (df_vid, channel_id):
    """
    Plot distribution of different video variables for a given channel.
    
    --------------------------------------------------------
    Parameters:
    df_vid (pl.Dataframe):
        video metadata dataframe with the data grouped by chanel over wich we want to compute the t test
        
    channel_id (string):
        id of the channel we want to plot
            
    --------------------------------------------------------
    Returns: None
        It only plots the variables, and does not return anything.
    """
    

    channel = df_vid.filter(pl.col('channel_id') == channel_id)
    
    if (len((df_vid['channel_id'] == channel_id).unique()) == 1):
        return "Invalid channel id"

    titles = ['dislike_count','duration','like_count','view_count', 'num_comms']
    xlabels = ['Number of dislikes per video','Length of the video [s]', 'Number of likes per video', 'Number of views per video', 'Number of comments per video']

    for i in range(len(titles)):
        plt.hist(channel[titles[i]])
        plt.title(titles[i])
        plt.xlabel(xlabels[i])
        plt.ylabel('Count')
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
    
def normalize_vids_with_timeseries(df_vid, df_timeseries, feature_to_divide_by):
    """
    Normalize video metadata by a specified feature from a timeseries dataset.
    This function divides all specified columns in the video metadata dataframe by a 
    desired feature from the timeseries dataset (e.g., number of subscribers, 
    delta views, etc.). It is also designed to work with the channel dataframe.
    
    --------------------------------------------------------
    Parameters:
    df_vid (pl.DataFrame):
        The dataframe containing video metadata.
        
    df_timeseries (pl.DataFrame):
        The dataframe containing timeseries data (or channel data).
    
    feature_to_divide_by (str):
        The feature from the timeseries dataframe to divide the video metadata by.
    --------------------------------------------------------
    Returns: DataFrame
        A new dataframe with the specified columns normalized by the given feature from the timeseries dataframe.
    """
    
    grouped_ts = df_timeseries.group_by('channel_id').mean()[['channel_id',feature_to_divide_by]]
    merged = df_vid.join(grouped_ts, on='channel_id')
    return merged.with_columns(pl.col(['dislike_count','duration','like_count','view_count','num_comms']) / merged[feature_to_divide_by])

def ttest_between_events(metric_event1, metric_event2):
    """
    Perform an independent t-test between two sets of event metrics.

    This function takes two sets of event metric, removes any null values,
    and performs an independent t-test to determine if there is a significant
    difference between the two sets.

    --------------------------------------------------------
    Parameters:
    metric_event1 (pl.Series): 
        The first set of event metrics.
    
    metric_event2 (pl.Series):
        The second set of event metrics.
    --------------------------------------------------------
    Returns (tuple):
        A tuple containing the t-statistic and the p-value of the test.
    """
    event1 = metric_event1.drop_nulls()
    event2 = metric_event2.drop_nulls()
    return stats.ttest_ind(event1,event2)

def Ftest_between_events (metric_event1, metric_event2):
    """
    Perform an F-test between two sets of event metrics.

    This function takes two sets of event metric, removes any null values,
    and performs an F-test to determine if there is a significant
    difference between the two sets.

    --------------------------------------------------------
    Parameters:
    metric_event1 (pl.Series): 
        The first set of event metrics.
    
    metric_event2 (pl.Series):
        The second set of event metrics.
    --------------------------------------------------------
    Returns (tuple):
        A tuple containing the t-statistic and the p-value of the test.
    """
    event1 = metric_event1.drop_nulls()
    event2 = metric_event2.drop_nulls()
    return stats.f_oneway(event1,event2)

def get_general_vid_statistics(filtered_df, cols_to_keep = ['dislike_count','duration','like_count','view_count','num_comms']):

    """
    Compute general statistics on the video metadata dataset.
    
    --------------------------------------------------------
    Parameters:
    filtered_df (pl.Dataframe):
        Video metadata dataframe over wich we want to compute the statistics.
        
    cols_to_keep : list of strings
        Columns over which we want to compute the statistics.
              
    --------------------------------------------------------
    Returns (tuple):
        A tuple containing :
       
        means (pl.Dataframe):
            Array with the mean of each metric for each channel.
        
        stdevs (pl.Dataframe)
            Array with the standard deviation of each metric for each channel.
            
        medians (pl.Dataframe)
            Array with the median of each metric for each channel.
    
    """

    means = filtered_df.mean()[cols_to_keep]
    stdevs = filtered_df.std()[cols_to_keep]
    medians = filtered_df.median()[cols_to_keep]
    
    return means,stdevs,medians

def compare_overall_vid_count_between_events (vid_count_1,vid_count_2):
    """
    Compare the overall video counts between two events channel-wise.
    This function prints the total number of videos for each event and the difference in video counts between the two events.
    It also returns a DataFrame that joins the two input DataFrames on the 'channel_id' column, renaming the 'counts' columns
    to 'counts_1' and 'counts_2' respectively.
    
    --------------------------------------------------------
    Parameters:
    vid_count_1 (pd.DataFrame): 
        A DataFrame containing video counts for event 1 with columns 'channel_id' and 'counts'.
        (Same format as the output of get_general_ch_statistics)
        
    vid_count_2 (pd.DataFrame): 
        A DataFrame containing video counts for event 2 with columns 'channel_id' and 'counts'.
        (Same format as the output of get_general_ch_statistics)
    
    --------------------------------------------------------    
    Returns (pl.DataFrame)
        A DataFrame with the video count for ach event forr all channels.
    """
    
    print ('Total number of videos for event 1 : ', vid_count_1['counts'].sum())
    print ('Total number of videos for event 2 : ', vid_count_2['counts'].sum())
    print ('Difference in videos from event 1 to event 2 : ', vid_count_1['counts'].sum() - vid_count_2['counts'].sum())
    return vid_count_1.join(vid_count_2, on='channel_id').rename({'counts':'counts_1', 'counts_right':'counts_2'})

def compare_video_statistics_between_events (videos_1, videos_2, metric = 'mean'):
    """
    Compare the overall video metrics between two events.
    This function prints the total number of videos for each event and the difference in video counts between the two events.
    It also returns a DataFrame that joins the two input DataFrames on the 'channel_id' column, renaming the 'counts' columns
    to 'counts_1' and 'counts_2' respectively.
    
    --------------------------------------------------------
    Parameters:
    videos_1 (pl.DataFrame): 
        A DataFrame containing video data for event 1.
        
    videos_2 (pl.DataFrame): 
        A DataFrame containing video data for event 2.
        
    metric (string):
        A string indicating which metric to anylize.
        Can take values in : ['mean','std','med']
    --------------------------------------------------------    
    Returns (pl.DataFrame)
        A DataFrame with metric calculate for each event and the difference between them.
    """
    
    v_means_1,v_stdevs_1,v_medians_1 = get_general_vid_statistics(videos_1)
    v_means_2,v_stdevs_2,v_medians_2 = get_general_vid_statistics(videos_2)
    
    if metric == 'mean' :
        return pl.concat([v_means_1,v_means_2,v_means_2-v_means_1]).insert_column(0,pl.Series(['event_1','event_2','difference']))
    
    if metric == 'std' :
        return pl.concat([v_stdevs_1,v_stdevs_2,v_stdevs_2-v_stdevs_1]).insert_column(0,pl.Series(['event_1','event_2','difference']))
    
    if metric == 'med' :
        return pl.concat([v_medians_1,v_medians_2,v_medians_2-v_medians_1]).insert_column(0,pl.Series(['event_1','event_2','difference']))
    
    else :
        return "Invalid metric, please enter one of the following options : ['mean','std','med']"

def plot_correlation_matrix (df, plot_title):
    """
    Plot the correlation matrix for a given dataframe.
    This works for the all three main datasets (channels, timeseries, video metadata)
    
    --------------------------------------------------------
    Parameters:
    df (pd.DataFrame): 
        The dataframe we want to plot.
        
    plot_title (string): 
        The title of the correlation matrix plot.
    
    --------------------------------------------------------    
    Returns:
    cov (pl.DataFrame)
        The covariance matrix for the given dataset
    """
    
    
    filtered_df = df.select(pl.col(pl.Float64, pl.Int64)).drop_nulls()
    
    for i in df.columns : 
        if i == '':
            filtered_df = filtered_df.drop('')
    
    df_standardized = stats.zscore(pd.DataFrame(filtered_df), axis=1)
    
    cov = df_standardized.corr()
    metrics = filtered_df.columns
    mask = np.triu(np.ones_like(cov, dtype=bool))

    plt.figure(figsize=(7, 5))
    fig = sns.heatmap(cov, mask=mask, center=0, annot=True,
                fmt='.2f', square=True, cmap='RdYlBu')

    fig.set_xticks(np.arange(len(metrics)) + 1/2, labels=metrics)
    fig.set_yticks(np.arange(len(metrics))+ 1/2, labels=metrics)

    plt.setp(fig.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(fig.get_yticklabels(), rotation=0, ha="right",rotation_mode="anchor")

    plt.title(plot_title)
    plt.show()
    
    return cov
    
    