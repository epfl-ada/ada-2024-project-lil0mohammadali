import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy import stats

def get_general_ch_statistics(filtered_df, cols_to_keep = ['dislike_count','duration','like_count','view_count'], channel = False):

    '''get general statistics for all channel on a given column'''

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
    
    '''ttest : checks the null hypothesis that two independant channels have an identical mean number of views, likes etc...
    used to compare if two sample's means differ significantly or not'''

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
    
    '''F test : test for the null hypothesis that two channels have the same variance
    used to compare if two sample's variance differ significantly or not'''
    
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
    '''plot distribution of different variables for a given channel'''

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
    
def normalize_vids_with_timeseries (df_vid, df_timeseries, feature_to_divide_by):
    
    '''Divide all of the data by the desired feature from the timeseries dataset (number of subs, delta views, ...)
    Note : it should also work for the channel dataframe'''
    
    grouped_ts = df_timeseries.group_by('channel_id').mean()[['channel_id',feature_to_divide_by]]
    merged = df_vid.join(grouped_ts, on='channel_id')
    return merged.with_columns(pl.col(['dislike_count','duration','like_count','view_count','num_comms']) / merged[feature_to_divide_by])

def ttest_between_events (statistic_event1, statistic_event2):
    event1 = statistic_event1
    event2 = statistic_event2
    event1 = event1.drop_nulls()
    event2 = event2.drop_nulls()
    return stats.ttest_ind(event1,event2)

def Ftest_between_events (statistic_event1, statistic_event2):
    event1 = statistic_event1
    event2 = statistic_event2
    event1 = event1.drop_nulls()
    event2 = event2.drop_nulls()
    return stats.f_oneway(event1,event2)

def get_general_vid_statistics(filtered_df, cols_to_keep = ['dislike_count','duration','like_count','view_count']):

    '''get general statistics for all channel on a given column'''

    means = filtered_df.mean()[cols_to_keep]
    stdevs = filtered_df.std()[cols_to_keep]
    medians = filtered_df.median()[cols_to_keep]
    
    return means,stdevs,medians

def compare_overall_vid_count_between_events (vid_count_1,vid_count_2):
    
    print ('Total number of videos for event 1 : ', vid_count_1['counts'].sum())
    print ('Total number of videos for event 2 : ', vid_count_2['counts'].sum())
    print ('Difference in videos from event 1 to event 2 : ', vid_count_1['counts'].sum() - vid_count_2['counts'].sum())
    return vid_count_1.join(vid_count_2, on='channel_id').rename({'counts':'counts_1', 'counts_right':'counts_2'})

def compare_video_statistics_between_events (videos_1, videos_2):
    
    v_means_1,v_stdevs_1,v_medians_1 = get_general_vid_statistics(videos_1)
    v_means_2,v_stdevs_2,v_medians_2 = get_general_vid_statistics(videos_2)
    
    print ('Average number of views for each video in event 1 : ', v_means_1['view_count'].sum())
    print ('Average number of views for each video in event 2 : ', v_means_2['view_count'].sum())
    print ('Difference average number of views for each video from event 1 to event 2 : ', v_means_1['view_count'].sum() - v_means_2['view_count'].sum())
    return pl.concat([v_means_1,v_means_2]).insert_column(0,pl.Series(['event_1','event_2']))

def plot_covariance (df, titlecov, titlehist):
    cov = np.cov(df.drop('channel_id').drop_nulls())
    plt.imshow(cov)
    plt.colorbar()
    plt.title(titlecov)
    plt.tight_layout()
    plt.show()

    #plot histogram of correlations to know how to evaluate the data.
    plt.hist(cov.flatten())
    plt.yscale('log')
    plt.title(titlehist)
    plt.ylabel('Log count')
    plt.xlabel('Covariance')
    plt.show()
    
    return cov

def get_correlated_channels (df, threshold):
    cov = np.cov(df.drop('channel_id').drop_nulls()) 
    
    first_correlated_channel = df[np.where(np.tri(len(cov), k=-1) * cov > threshold)[0]]
    second_correlated_channel = df[np.where(np.tri(len(cov), k=-1) * cov > threshold)[1]]

    correlated_channels = first_correlated_channel.insert_column(1,second_correlated_channel[:,0].rename('channel_id_2'))[['channel_id','channel_id_2']]
    return correlated_channels.rename({'channel_id':'channel_id_1'})