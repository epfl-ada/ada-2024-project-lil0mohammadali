import pandas as pd
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from nltk.corpus import stopwords #needs 'pip install nltk'
import nltk
nltk.download('stopwords')

def plot_video_stat(video_list, stat):
    """plots histogramms of number based video stats

    Parameters
    ----------
    video_list : pd.df
        the data frame with the videos that are to be plotted
    stat : str
        name of the stat that we want to plot. ex: 'view_count'
    """
    if isinstance(video_list, pl.DataFrame): # if polars convert to pandas 
        video_stat = video_list[stat].to_pandas() #needs "pip install pyarrow" to run
    else:
        video_stat = video_list[stat]
    num_stat = pd.to_numeric(video_stat, errors='coerce')
    if stat =='duration':
        num_stat= num_stat/60/60 #convert from seconds to minutes
        print("12 hours is the limit for livestream uploads")
        stat = 'Duration in Hours'
    plt.hist(num_stat, bins= 100, edgecolor='black')
    plt.xlabel('Video '+stat)
    plt.ylabel('Frequency')
    plt.title('Distribution of Video '+stat)
    plt.yscale('log')
    plt.show()

def plot_most_common_words(video_list, text, topX):
    """plots the most common words in text based video parameters

    Parameters
    ----------
    video_list : pd.df
        the data frame with the videos that are to be plotted
    text : str
        name of the text that we want to plot. ex: 'description'
    topX : int
        number of words to plot. ex: 10 to get the 10 most common words
    """
    if isinstance(video_list, pl.DataFrame): # if polars convert to pandas 
        video_text = video_list[text].to_pandas() 
    else:
        video_text = video_list[text]
    video_text= str.split(video_text.to_string(index=False))
    video_text= pd.Series(video_text)
    #filtering
    stop_words = set(stopwords.words('english'))
    filtered = [token for token in video_text if token.lower() not in stop_words]
    filtered = pd.Series(filtered)
    filtered = filtered.str.replace('.','')
    filtered = filtered.str.lower() #remove duplicates with different capitalisation
    filtered = filtered[filtered.str.len() > 1] #remove prepostions
    top_words = filtered.value_counts().head(topX)
    plt.figure()
    plt.xlabel(str(topX)+' most common words in video '+text)
    plt.ylabel('Frequency')
    top_words.plot(kind='bar', figsize=(20, 8))
    plt.yscale('log')
    plt.title('Counts of the '+str(topX)+' most common words in video '+text+' excluding stopwords')
    plt.show()

def plot_most_common_tags(video_list, topX):
    """plots the most common tags of the videos

    Parameters
    ----------
    video_list : pd.df
        the data frame with the videos that are to be plotted
    topX : int
        number of tags to plot. ex: 10 to get the 10 most common tags
    """
    if isinstance(video_list, pl.DataFrame): # if polars convert to pandas 
        video_text = video_list['tags'].to_pandas() 
    else:
        video_text = video_list['tags']
    video_text= str.split(video_text.to_string(index=False), sep=',')
    video_text= pd.Series(video_text)
    #filtering
    filtered = video_text.str.lower() #remove duplicates with different capitalisation
    top_words = filtered.value_counts().head(topX)
    plt.figure()
    plt.xlabel(str(topX)+' most common tags')
    plt.ylabel('Frequency')
    top_words.plot(kind='bar', figsize=(20, 8))
    plt.yscale('log')
    plt.title('Counts of the '+str(topX)+' most common words in video tags')
    plt.show()

def plot_text_len_char(video_list, text):
    """plots the distribution of text length measured in number of characters

    Parameters
    ----------
    video_list : pd.df
        the data frame with the videos that are to be plotted
    text : str
        name of the text that we want to plot. ex: 'title'
    """
    if isinstance(video_list, pl.DataFrame): # if polars convert to pandas 
        video_text = video_list[text].to_pandas() 
    else:
        video_text = video_list[text]
    text_len= video_text.str.len()
    plt.figure()
    plt.hist(text_len, bins= 100, edgecolor='black')
    plt.xlabel('Video '+text+'length in number of charaters')
    plt.ylabel('Frequency')
    plt.title('distribution of '+text+' length')
    if text=='description':
        plt.yscale('log')
    plt.show()

def plot_text_len_words(video_list, text):
    """plots the distribution of text length measured in number of characters

    Parameters
    ----------
    video_list : pd.df
        the data frame with the videos that are to be plotted
    text : str
        name of the text that we want to plot. ex: 'title'
    """
    if isinstance(video_list, pl.DataFrame): # if polars convert to pandas 
        video_text = video_list[text].to_pandas() 
    else:
        video_text = video_list[text]    
    video_text = video_text.astype(str)
    word_count = video_text.apply(lambda x:str.split(x))
    word_count = word_count.apply(lambda x: len(x))
    plt.figure()
    plt.hist(word_count, bins= 100, edgecolor='black')
    plt.xlabel('Video '+text+' length in number of words')
    plt.ylabel('Frequency')
    if text=='description':
        plt.yscale('log')
    plt.title('Counts of words in Video '+text)
    plt.show()

def capitalisation_ratio(text):
    """returns the ratio of upper case letters to lower case ones

    Parameters
    ----------
    text : str
        the text that we want to analze

    Returns
    ----------
    ratio : float
        the ratio of capitalization
    """
    up_count = sum(1 for c in text if c.isupper())
    low_count = sum(1 for c in text if c.islower())
    #handle edge cases
    if low_count == 0:
        return float('inf') if up_count > 0 else 0  
    return up_count / low_count


def cap_ratio(video_list, text):
    """returns the ratio of UPPER CASE/lower case letter

    Parameters
    ----------
    video_list : pd.df
        the data frame with the videos that are to be plotted
    text : str
        name of the text that we want to analyze. ex: 'title'

     Returns
    ----------
    ratios : pd.df
        a df column that corresponds the the capitalisation ratios   
    """
    video_list = video_list.with_columns(
    pl.col(text).map_elements(capitalisation_ratio).alias("capitalisation_ratio"))
    return video_list

def highperformer(video_list, category, percentage=5):
    """returns the list of videos in the upper x percentage in a category

    Parameters
    ----------
    video_list : pd.df
        the list of videos to analyse
    category : str
        the category in which to rank the videos. ex: 'view_count'
    percentage : int
        the perchentage of videos to retrun. ex: 5 resturns the top 5% of videos

    Returns
    ----------
    high_perf : pd.df
        video metadate ranked by the category and with in the top range
    """
    video_list = video_list.with_columns(
        pl.col(category).cast(pl.Float64, strict=False)  
    )
    high_perf = video_list.sort(category, descending=True)
    num_videos = video_list.height
    top_n = int(round(num_videos * percentage / 100))
    return high_perf.head(top_n)

def get_general_ch_statistics(filtered_df, cols_to_keep = ['dislike_count','duration','like_count','view_count','num_comms'], channel = False):

    """
    Group video metadata dataset by channel id and compute some general statistics.
    
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
    Perform a t-test a given variable between two different channels to asses the independence of their means.
    
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
    Perform an F-test for a given variable between two different channels to asses the independence of their variances.
    
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

def ttest (metric_1, metric_2):
    """
    Perform an independent t-test between two sets of metrics.

    This function takes two sets of metric, removes any null values,
    and performs an independent t-test to determine if there is a significant
    difference between the two sets.
    
    Can be used to compare between different events, or to link two different variables.

    --------------------------------------------------------
    Parameters:
    metric_event1 (pl.Series): 
        The first set of metrics.
    
    metric_event2 (pl.Series):
        The second set of metrics.
    --------------------------------------------------------
    Returns (tuple):
        A tuple containing the t-statistic and the p-value of the test.
    """
    cleaned1 = metric_1.drop_nulls()
    cleaned2 = metric_2.drop_nulls()
    return stats.ttest_ind(cleaned1,cleaned2)

def Ftest (metric_1, metric_2):
    """
    Perform an F-test between two sets of metrics.

    This function takes two sets of metrics, removes any null values,
    and performs an F-test to determine if there is a significant
    difference between the two sets.
    
    Can be used to compare between different events, or to link two different variables.

    --------------------------------------------------------
    Parameters:
    metric_event1 (pl.Series): 
        The first set of metrics.
    
    metric_event2 (pl.Series):
        The second set of metrics.
    --------------------------------------------------------
    Returns (tuple):
        A tuple containing the t-statistic and the p-value of the test.
    """
    cleaned1 = metric_1.drop_nulls()
    cleaned2 = metric_2.drop_nulls()
    return stats.f_oneway(cleaned1,cleaned2)

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
    
def plot_correlation_matrix_features_and_metrics (features_and_metrics, shift_lines = 8):
    """
    Plot the correlation matrix for the features and metrics dataframes.
    
    --------------------------------------------------------
    Parameters:
    features and metrics (pd.DataFrame): 
        The dataframe we want to plot.
    
    shift_lines (int):
        shifts the black lines separating metrics from features
    
    --------------------------------------------------------    
    Returns:
    cov (pl.DataFrame)
        The covariance matrix for the given dataset
    """
    
    filtered_df = features_and_metrics.select(pl.col(pl.Float64, pl.Int64)).drop_nulls()
        
    for i in features_and_metrics.columns : 
        if i == '':
            filtered_df = filtered_df.drop('')
        
    df_standardized = stats.zscore(pd.DataFrame(filtered_df), axis=1)
        
    cov = df_standardized.corr()
    metrics = filtered_df.columns
    mask = np.triu(np.ones_like(cov, dtype=bool))

    plt.figure(figsize=(10, 7))
    fig = sns.heatmap(cov, mask=mask, center=0, annot=True,
                fmt='.2f', square=True, cmap='RdYlBu')

    fig.set_xticks(np.arange(len(metrics)) + 1/2, labels=metrics)
    fig.set_yticks(np.arange(len(metrics)) + 1/2, labels=metrics)

    fig.hlines(shift_lines, *fig.get_xlim(), colors='black')
    fig.vlines(shift_lines, *fig.get_ylim(), colors='black')

    plt.setp(fig.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(fig.get_yticklabels(), rotation=0, ha="right",rotation_mode="anchor")

    plt.title('Correlation matrix of features and metrics')
    plt.show()
    
    return cov


def comment_replies_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes the average, median, 25% and 75% quartile of the replies per video
    Args:
        - df: The comments dataset
    """
    assert df.columns == ['author', 'video_id', 'likes', 'replies'], "please provide "\
        "dataset with the columns ['author', 'video_id', 'likes', 'replies']"
    grouped_df = df.group_by(by="video_id").agg(pl.col('replies').mean().name.suffix("_mean"), 
                                        pl.col('replies').count().name.suffix("_num"), 
                                        pl.col('replies').median().name.suffix("_median"),
                                        pl.col('replies').quantile(0.25, interpolation = "midpoint").name.suffix("_25"),
                                        pl.col('replies').quantile(0.75, interpolation = "midpoint").name.suffix("_75"),
                                        )
    grouped_df = grouped_df.rename({"by": "video_id"})
    return grouped_df

def filter_by_date(df, col_name, min_date, max_date):
    """
    Filter a given dataframe by date.
    
    --------------------------------------------------------
    Parameters:
    df (pl.DataFrame):
        dataframe we want to filter
    
    col_name (string):
        name of the column we want to filter by
    
    min_date (pl.datetime):
        minimum date
        
    max_date (pl.datetime):
        maximum date
    
    --------------------------------------------------------    
    Returns:
    filtered_df (pl.DataFrame)
        The filtered dataframe.
    """
    
    
    string_to_date = df.with_columns(pl.col(col_name).str.to_datetime)
    filtered_df = string_to_date.filter((pl.col(col_name) >= min_date) & (pl.col(col_name) <= max_date))
    return filtered_df