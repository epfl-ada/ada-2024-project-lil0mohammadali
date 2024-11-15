import pandas as pd
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

def plot_video_stat(filtered_meta, stat):
    video_stat = filtered_meta[stat].dropna()
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

def plot_most_common_words(filtered_meta, text, topX):
    video_text = filtered_meta[text]
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

def plot_most_common_tags(filtered_meta, topX):
    video_text = filtered_meta['tags']
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

def plot_text_len_char(filtered_meta, text):
    video_text = filtered_meta[text]
    text_len= video_text.str.len()
    plt.figure()
    plt.hist(text_len, bins= 100, edgecolor='black')
    plt.xlabel('Video '+text+'length in number of charaters')
    plt.ylabel('Frequency')
    plt.title('distribution of '+text+' length')
    if text=='description':
        plt.yscale('log')
    plt.show()

def plot_text_len_words(meta, text):
    video_text = meta[text]
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
    up_count = sum(1 for c in text if c.isupper())
    low_count = sum(1 for c in text if c.islower())
    #handle edgecases
    if low_count == 0:
        return float('inf') if up_count > 0 else 0  
    return up_count / low_count

def highperformer(video_list, category, percentage=5):
    video_list[category] = pd.to_numeric(video_list[category], errors='coerce')
    high_perf = video_list.sort_values(by=category, ascending=False)
    num_videos = len(video_list)
    return(high_perf.head(int(round(num_videos*percentage/100))))

