import pandas as pd
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

# def highperformers(video_list, category, percentage):
