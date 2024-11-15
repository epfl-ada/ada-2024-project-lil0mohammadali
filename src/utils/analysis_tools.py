import pandas as pd
import polars as pl
import numpy as np
from matplotlib import pyplot as plt

def plot_video_stat(video_list, stat):
    """plots histogramms of number based video stats

    Parameters
    ----------
    video_list : pd.df
        the data frame with the videos that are to be plotted
    stat : str
        name of the stat that we want to plot. ex: 'view_count'
    """
    video_stat = video_list[stat].dropna()
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
    video_text = video_list[text]
    video_text= str.split(video_text.to_string(index=False))
    video_text= pd.Series(video_text)
    #filtering
    filtered = video_text[video_text.str.len() > 3] #remove prepostions
    filtered = filtered.str.lower() #remove duplicates with different capitalisation
    top_words = filtered.value_counts().head(topX)
    plt.figure()
    plt.xlabel(str(topX)+' most common words in video '+text)
    plt.ylabel('Frequency')
    top_words.plot(kind='bar', figsize=(20, 8))
    plt.yscale('log')
    plt.title('Counts of the '+str(topX)+' most common words in video '+text+' with more than 3 letters')
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
    #handle edgecases
    if low_count == 0:
        return float('inf') if up_count > 0 else 0  
    return up_count / low_count

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
    video_list[category] = pd.to_numeric(video_list[category], errors='coerce')
    high_perf = video_list.sort_values(by=category, ascending=False)
    num_videos = len(video_list)
    return(high_perf.head(int(round(num_videos*percentage/100))))


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