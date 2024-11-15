import pandas as pd
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


def plot_video_stat(filtered_meta, stat):
    if isinstance(filtered_meta, pl.DataFrame): # if polars convert to pandas 
        video_stat = filtered_meta[stat].to_pandas() #needs "pip install pyarrow" to run
    else:
        video_stat = filtered_meta[stat]
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
    if isinstance(filtered_meta, pl.DataFrame): # if polars convert to pandas 
        video_text = filtered_meta[text].to_pandas()
    else:
        video_text = filtered_meta[text]
    video_text= str.split(video_text.to_string(index=False))
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
    plt.title('Counts of the '+str(topX)+' most common words in video '+text+' with more than 3 letters')
    plt.show()

def plot_most_common_tags(filtered_meta, topX):
    if isinstance(filtered_meta, pl.DataFrame): # if polars convert to pandas 
        video_text = filtered_meta['tags'].to_pandas()
    else:
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
    if isinstance(filtered_meta, pl.DataFrame): # if polars convert to pandas 
        video_text = filtered_meta[text].to_pandas()
    else:
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
    if isinstance(meta, pl.DataFrame): # if polars convert to pandas 
        video_text = meta[text].to_pandas()
    else:
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
    #handle edge cases
    if low_count == 0:
        return float('inf') if up_count > 0 else 0  
    return up_count / low_count

def cap_ratio(video_list, text):
    video_list = video_list.with_columns(
    pl.col(text).map_elements(capitalisation_ratio).alias("capitalisation_ratio")
    )
    return video_list

def highperformer(video_list, category, percentage=5):
    video_list = video_list.with_columns(
        pl.col(category).cast(pl.Float64, strict=False)  
    )
    high_perf = video_list.sort(category, descending=True)
    num_videos = video_list.height
    top_n = int(round(num_videos * percentage / 100))
    return high_perf.head(top_n)


