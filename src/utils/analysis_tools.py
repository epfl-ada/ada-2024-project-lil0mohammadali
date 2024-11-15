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
    video_t= pd.Series(video_text)
    #filtering to remove prepostitions
    filtered = video_text[video_text.str.len() > 3]
    filtered = filtered.str.lower()
    top_words = filtered.value_counts().head(topX)
    plt.figure()
    top_words.plot(kind='bar', figsize=(20, 8))
    plt.yscale('log')
    plt.title('Counts of words in video '+text+' with more than 10 000 occurrences and more than 3 letters')
    plt.show()

def plot_text_len_char(filtered_meta, text):
    video_text = filtered_meta[text]
    text_len= video_text.str.len()
    plt.figure()
    text_len.value_counts().plot(kind='hist', bins=300, figsize=(20, 8))
    plt.title('distribution of '+text+' length in number of characters')
    if text=='description':
        plt.yscale('log')
    plt.show()

def plot_text_len_words(meta, text):
    video_text = meta[text]
    video_text = video_text.astype(str)
    word_count = video_text.apply(lambda x:str.split(x))
    word_count = word_count.apply(lambda x: len(x))
    plt.figure()
    word_count.value_counts().plot(kind='hist', bins = 300, figsize=(20, 8))
    if text=='description':
        plt.yscale('log')
    plt.title('Counts of words in Video '+text)
    plt.show()
