import pandas as pd
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from nltk.corpus import stopwords #needs 'pip install nltk'
import nltk
nltk.download('stopwords')
from src.utils.keywords import add_video_live
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    nltk.download('stopwords', quiet=True)
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
    if (low_count + up_count) == 0:
        return np.nan if up_count > 0 else 0  
    return up_count / (low_count+up_count)


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
    pl.col(text).map_elements(capitalisation_ratio, return_dtype = pl.Float64).alias("capitalisation_ratio"))
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

def ttest_matrix(df, plot=False, plot_title='Matrix of p-values for t-tests between features'):
    '''
    Compute a matrix of p-values for t-tests between the columns of a dataframe
    
    Parameters:
    df: pd.DataFrame
        The dataframe for which to compute the t-test matrix
    plot: bool
        Whether to plot the matrix of p-values
    plot_title: str
        The title of the plot
    
    Returns:
    ttests: np.array
        The matrix of p-values for t-tests between the columns of the dataframe
    '''
    n_features = df.shape[1]
    ttests = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            ttests[i,j] = stats.ttest_ind(df[:,i], df[:,j], equal_var=False)[1]
    
    if plot:
        labels = df.columns
        mask = np.triu(np.ones_like(ttests, dtype=bool))
        
        plt.figure(figsize=(10, 7))
        fig = sns.heatmap(ttests, mask=mask, center=0, annot=True, fmt='.2f', square=True, cmap='RdYlBu')
        fig.set_xticks(np.arange(len(labels)) + 1/2, labels=labels)
        fig.set_yticks(np.arange(len(labels)) + 1/2, labels=labels)
        plt.setp(fig.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        plt.setp(fig.get_yticklabels(), rotation=0, ha="right",rotation_mode="anchor")

        plt.title(plot_title)
        plt.show()
    return ttests

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

def compute_correlation_matrix (df, how = 'pearson', plot = False, plot_title = 'Correlation Matrix'):
    """
    Compute the correlation matrix for a given dataframe.
    This works for the all three main datasets (channels, timeseries, video metadata)
    
    --------------------------------------------------------
    Parameters:
    df (pd.DataFrame): 
        The dataframe we want to plot.
        
    how (str):
        The correlation coefficient to compute. Can be 'pearson' or 'spearman'. (default is 'pearson')
        The pearson coefficient is used to measure the linear correlation between two variables.
        The spearman coefficient is used to measure the monotonic relationship between two variables.
    
    plot (bool):
        If True, the correlation matrix is plotted. If False, only the matrix is returned.
        
    plot_title (string): 
        The title of the correlation matrix plot.
    
    
    --------------------------------------------------------    
    Returns:
    corr (np.array)
        The correlation matrix for the given dataset
        
    pvals (np.array)
        The p-values for the correlation matrix
    """
    
    
    filtered_df = df.select(pl.col(pl.Float64, pl.Int64))
    
    if any('is_live' in s for s in df.columns):
        if  len(filtered_df['is_live'].unique()) == 1:
            filtered_df = filtered_df.drop('is_live')
    
    df_standardized = stats.zscore(pd.DataFrame(filtered_df), axis=0)
    
    if how == 'pearson':
        corr = np.ones((df_standardized.shape[1], df_standardized.shape[1]))
        pvals = np.ones((df_standardized.shape[1], df_standardized.shape[1]))
        for i in range(df_standardized.shape[1]):
            for j in range(df_standardized.shape[1]):
                corr[i,j] = stats.pearsonr(df_standardized[i], df_standardized[j])[0]
                pvals[i,j] = stats.pearsonr(df_standardized[i], df_standardized[j])[1]
    
    if how == 'spearman':
        corr, pvals = stats.spearmanr(df_standardized.to_numpy())
    
    if plot:
        labels = filtered_df.columns
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(7, 5))
        fig = sns.heatmap(corr, mask=mask, center=0, annot=True,
                    fmt='.2f', square=True, cmap='RdYlBu')

        fig.set_xticks(np.arange(len(labels)) + 1/2, labels=labels)
        fig.set_yticks(np.arange(len(labels))+ 1/2, labels=labels)

        plt.setp(fig.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        plt.setp(fig.get_yticklabels(), rotation=0, ha="right",rotation_mode="anchor")

        plt.title(plot_title)
        plt.show()
    
    return corr, pvals
    
def compute_correlation_matrix_features_and_metrics (vid_features, response_metrics, how='pearson'):
    """
    Compute the correlation matrix for the features and metrics dataframes, and plot it as a heatmap.
    
    --------------------------------------------------------
    Parameters:
    vid_features (pd.DataFrame):
        The dataframe containing the video features.
    metrics (pd.DataFrame): 
        The dataframe containing the response metrics.
    how (str):
        The correlation coefficient to compute. Can be 'pearson' or 'spearman'. (default is 'pearson')
        The pearson coefficient is used to measure the linear correlation between two variables.
        The spearman coefficient is used to measure the monotonic relationship between two variables.
    
    --------------------------------------------------------    
    Returns:
    corr (numpy array)
        The correlation matrix for the given dataset
    pvals (numpy array)
        The p-values for the correlation matrix
    """
    
    features_and_metrics = vid_features.join(response_metrics, on='display_id', how='inner')
    
    filtered_df = features_and_metrics.select(pl.col(pl.Float64, pl.Int64))
    
    if any('is_live' in s for s in filtered_df.columns):
        filtered_df = filtered_df.drop('is_live')
    
    df_standardized = stats.zscore(pd.DataFrame(filtered_df), axis=0)
    
    if how == 'pearson':
        corr = np.ones((df_standardized.shape[1], df_standardized.shape[1]))
        pvals = np.ones((df_standardized.shape[1], df_standardized.shape[1]))
        for i in range(df_standardized.shape[1]):
            for j in range(df_standardized.shape[1]):
                corr[i,j], pvals[i,j] = stats.pearsonr(df_standardized[i], df_standardized[j])
    
    if how == 'spearman':
        corr, pvals = stats.spearmanr(df_standardized.to_numpy())
    
    return corr, pvals
    

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

def plot_pvals_for_correlation(pvals, list_vid_features, list_response_metrics, plot_full = False, plot_title = 'P-values of the correlation matrix of features and metrics'):
    """
    Plot the p-values for the correlation matrix between features and metrics.
    
    --------------------------------------------------------
    Parameters:
    vid_features (pl.DataFrame):
        The dataframe containing the video features (needed for labels).
    response_metrics (pl.DataFrame):
        The dataframe containing the response metrics (needed for labels).
    pvals (np.array):
        The p-values for the correlation matrix.
    plot_full (bool):
        If True, the full correlation matrix is plotted. If False, only the correlation between metrics and features is plotted.
        
    --------------------------------------------------------
    Returns: None
        It only plots the p-values, and does not return anything.
    """
    
    labels = list_vid_features + list_response_metrics
    
    labels = list(filter(lambda x: x != 'is_live' and x != 'display_id', labels))
    
    feature_metric_index_shift = len(list(filter(lambda x: x != 'is_live' and x != 'display_id', list_vid_features)))
    
    if plot_full :
        mask = np.triu(np.ones_like(pvals, dtype=bool))
        
        plt.figure(figsize=(10, 7))
        fig = sns.heatmap(pvals, mask=mask, annot=True, fmt='.2f', square=True, cmap='YlGnBu')
        
        fig.hlines(feature_metric_index_shift, *fig.get_xlim(), colors='black')
        fig.vlines(feature_metric_index_shift, *fig.get_ylim(), colors='black')
        
        fig.set_xticks(np.arange(len(labels)) + 1/2, labels=labels)
        fig.set_yticks(np.arange(len(labels)) + 1/2, labels=labels)

    else:     
        plt.figure(figsize=(10, 7))
        fig = sns.heatmap(pvals[feature_metric_index_shift:, :feature_metric_index_shift], annot=True, fmt='.2f', square=True, cmap='YlGnBu')

        fig.set_xticks(np.arange(len(labels[:feature_metric_index_shift])) + 1/2, labels=labels[:feature_metric_index_shift])
        fig.set_yticks(np.arange(len(labels[feature_metric_index_shift:])) + 1/2, labels=labels[feature_metric_index_shift:])
        
        fig.set_xlabel('Features')
        fig.set_ylabel('Metrics')
    

    plt.setp(fig.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(fig.get_yticklabels(), rotation=0, ha="right",rotation_mode="anchor")

    plt.title(plot_title)
    plt.show()
    
def plot_correlation_matrix_features_and_metrics(corr, list_vid_features, list_response_metrics, plot_full = False, plot_title = "Correlation matrix of features and metrics"):
    """
    Plot the correlation matrix between features and metrics.
    
    --------------------------------------------------------
    Parameters:
    vid_features (pl.DataFrame):
        The dataframe containing the video features (needed for labels).
    response_metrics (pl.DataFrame):
        The dataframe containing the response metrics(needed for labels).
    corr (np.array):
        The the correlation matrix calculated with compute_correlation_matrix_features_and_metrics.
    plot_full (bool):
        If True, the full correlation matrix is plotted. If False, only the correlation between metrics and features is plotted.
    plot_title (str):
        The title of the plot.
        
    --------------------------------------------------------
    Returns: None
        It only plots the matrix, and does not return anything.
    """
      
    labels = list_vid_features + list_response_metrics
    
    labels = list(filter(lambda x: x != 'is_live' and x != 'display_id', labels))
    
    feature_metric_index_shift = len(list(filter(lambda x: x != 'is_live' and x != 'display_id', list_vid_features)))
    
    if plot_full :
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        plt.figure(figsize=(10, 7))
        fig = sns.heatmap(corr, mask=mask, center=0, annot=True, fmt='.2f', square=True, cmap='RdYlBu')
        
        fig.hlines(feature_metric_index_shift, *fig.get_xlim(), colors='black')
        fig.vlines(feature_metric_index_shift, *fig.get_ylim(), colors='black')
        
        fig.set_xticks(np.arange(len(labels)) + 1/2, labels=labels)
        fig.set_yticks(np.arange(len(labels)) + 1/2, labels=labels)

    else:     
        plt.figure(figsize=(10, 7))
        fig = sns.heatmap(corr[feature_metric_index_shift:, :feature_metric_index_shift], center=0, annot=True, fmt='.2f', square=True, cmap='RdYlBu')

        fig.set_xticks(np.arange(len(labels[:feature_metric_index_shift])) + 1/2, labels=labels[:feature_metric_index_shift])
        fig.set_yticks(np.arange(len(labels[feature_metric_index_shift:])) + 1/2, labels=labels[feature_metric_index_shift:])
        
        fig.set_xlabel('Features')
        fig.set_ylabel('Metrics')

    plt.setp(fig.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(fig.get_yticklabels(), rotation=0, ha="right",rotation_mode="anchor")

    plt.title(plot_title)
    plt.show()

def compute_channel_activity(metadata: pl.DataFrame, channel_id: str, date:str, timespan:int):
    """Compute the channel activity for a given date and timespan
    
    Parameters:
    metadata: pl.DataFrame
        The metadata for the channels containing at least the columns 'channel_id', 'upload_date'
    channel_id: str
        The channel id for which to compute the activity
    date: str 
        The date for which to compute the activity in the format 'YYYY-MM-DD'
    timespan: int
        The timespan in days for which to compute the activity in +/- days around the date
    """

    assert 'channel_id' in metadata.columns, "The metadata should contain the column 'channel_id'"
    assert 'upload_date' in metadata.columns, "The metadata should contain the column 'upload_date'"

    channel_metadata = metadata.filter(pl.col('channel_id') == channel_id)
    date_list = pd.date_range(start = pd.to_datetime(date) - pd.DateOffset(days=timespan),
                                end = pd.to_datetime(date) + pd.DateOffset(days=timespan)).strftime('%Y-%m-%d').tolist()
    nb_videos = len(channel_metadata.filter(pl.col('upload_date').is_in(date_list)))
    return nb_videos/len(date_list)

def get_channels_activity(event_metadata: pl.DataFrame, feather_metadata: pl.DataFrame) -> pl.DataFrame:
    """Compute the activity for each channel in the videos related to the events

    Parameters:
    event_metadata: pl.DataFrame
        The metadata for the videos related to the events
    feather_metadata: pl.DataFrame
        The metadata for the channels containing at least the columns 'channel_id', 'upload_date'
    """
    
    assert 'channel_id' in feather_metadata.columns, "The metadata should contain the column 'channel_id'"
    assert 'upload_date' in feather_metadata.columns, "The metadata should contain the column 'upload_date'"

    activities = []
    index = 1
    num_rows = event_metadata.height
    for vid in event_metadata.iter_rows(named=True):
        print(f"Processing video {index}/{num_rows}\r", end="")
        index += 1
        activities.append(compute_channel_activity(feather_metadata, vid["channel_id"], 
                                                   vid['upload_date'], 14))
    return pl.DataFrame({
        'display_id': event_metadata['display_id'],
        'channel_activity': activities
    })

def create_video_features_dataframe (event_metadata, feather_metadata):
    """
    This function creates a dataframe containing features for each video in the event_metadata dataframe.
    The features are computed from the timeseries dataframe, and include the mean, standard deviation and median of the number of videos uploaded per day for the channel of each video.
    The function also computes the capitalization ratio of the title of each video, which is the number of capital letters divided by the number of lowercase letters in the title.
    
    Parameters:
    event_metadata: polars.DataFrame
        The dataframe containing the metadata of the videos in the event
    feather_metadata: polars.DataFrame
        The dataframe containing the metadata of ALL the videos from the channels
        
    Returns:
    vid_features: polars.DataFrame
        The dataframe containing the features for each video in the event_metadata dataframe
    """
    
    #select the relevant columns to create the vid_features dataframe
    vid_features = event_metadata[['display_id','channel_id', 'event_type', 'region', 'event',  'duration', 'is_live']]

    activity = get_channels_activity(event_metadata, feather_metadata)
    vid_features = vid_features.join(activity, on='display_id', how='inner')

    vid_features = vid_features.drop('channel_id')
    
    #compute the capitalization ratio of the title for each video
    titles = event_metadata[['display_id','title']]
    ratio = cap_ratio(titles, 'title')
    ratio = ratio.drop('title')
    vid_features = vid_features.join(ratio, on='display_id', how='inner')
    
    return vid_features

def create_response_metrics_df (event_metadata, num_comments):
    """
    This function creates a dataframe containing the public response metrics for each video in the event_metadata dataframe.
    The public response metrics are the number of likes, dislikes, views, comments and the ratio of likes to dislikes.
    The function also removes the rows with infinite like/dislike ratio and sets the NaN values to 1.
    
    Parameters:
    event_metadata: polars.DataFrame
        The dataframe containing the metadata of the videos in the event
    num_comments: polars.DataFrame
        The dataframe containing the number of comments for each video in the event
        
    Returns:
    response_metrics: polars.DataFrame
        The dataframe containing the public response metrics for each video in the event_metadata dataframe
    """

    #join the metadata with the number of comments to create the response metrics dataframe
    response_metrics = event_metadata[['display_id','like_count','dislike_count','view_count']].join(num_comments, on='display_id', how='inner')

    #deal with importing issues with the 'view_count' column
    response_metrics = response_metrics.with_columns(new_view_count = pl.col('view_count').str.strip_suffix('\r').str.strip_suffix('.0').cast(pl.Int64))
    response_metrics = response_metrics.drop('view_count').rename({'new_view_count':'view_count'})

    #add a column for likes/dislikes ratio
    response_metrics = response_metrics.with_columns((pl.col('like_count')/pl.col('dislike_count')).alias('likes/dislikes'))
    #drop the like_count and dislike_count columns since we have the ratio
    response_metrics = response_metrics.drop('like_count','dislike_count')

    #remove the entries with infinite like/dislike ratio, and set the NaN values to 1 (because they correspond to videow with no likes and no dislikes)
    response_metrics = response_metrics.with_columns(pl.col('likes/dislikes').replace(np.NaN, 1))
    response_metrics = response_metrics.filter(pl.col('likes/dislikes') != np.inf)
    
    return response_metrics


def plot_correlation_for_groups_of_events(list_of_correlations, list_of_pvalues, x_labels, y_labels, list_of_events): 

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Correlation Coefficients", "p-values"], horizontal_spacing = 0.2)

    fig.update_layout(autosize=False,title="Correlation Matrix of Video Features and Response Metrics", title_x=0.5, title_y=0.9, height=600, width=1400)


    buttons = []

    for i in range(len(list_of_events)):
        corr = list_of_correlations[i][len(x_labels):,:len(x_labels)]
        pvals = list_of_pvalues[i][len(x_labels):,:len(x_labels)]
        pvals_significant = (pvals < 0.05)

        fig.add_trace(go.Heatmap(z=corr, x=x_labels, y=y_labels, 
                                colorscale='RdYlBu', zmid = 0, zmin=-1, zmax=1,
                                colorbar={"title":"Correlation <br> coefficient", 'x':0.43}, 
                                hovertemplate = " Response metric : %{y}<br> Feature : %{x}<br> Corelation coefficient: %{z:.2f}<extra></extra>"),
                                row=1, col=1)
        
        fig.add_trace(go.Heatmap(z=pvals, x=x_labels, y=y_labels, 
                                colorscale= [
                                [0, 'rgb(250, 50, 50)'],        #0
                                [0, 'rgb(250, 170, 90)'],        #0
                                [1./1000, 'rgb(250, 250, 0)'],  #100
                                [1./100, 'rgb(160, 200, 250)'],  #1000
                                [1./10, 'rgb(50, 50, 250)'],       #10000

                                ], zmid=0.05, zmin=0, zmax=0.1,
                                colorbar={"title":"p-value", 'tickvals': [0, 0.025, 0.05, 0.075, 0.1],},
                                customdata = pvals_significant,
                                hovertemplate = 'Response metric : %{y}<br>' + 'Feature : %{x}<br>' + '<b>p-value : %{z:.2f}</b><br>'+ '<b>Significant : %{customdata}</b> <extra></extra>',
                                showlegend = False),
                                row=1, col=2)  

        buttons.append(dict(args=[{'visible': [False]*i*2+ [True]*2 + [False]*(len(list_of_events)-i-1)*2}], label=list_of_events[i], method="update"))

    initial_visibility = [True]*2 + [False]*(len(list_of_events)-1)*len(list_of_events)*2
    for i in range(len(fig.data)):
        fig.data[i].visible = initial_visibility[i]   


    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list(buttons),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )
        ],
    )

    fig.show(scrollZoom=False)