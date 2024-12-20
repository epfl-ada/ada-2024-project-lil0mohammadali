import pandas as pd
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import importlib

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import base64

from wordcloud import WordCloud, STOPWORDS



#bakes the images into the html
def encode_image(filename):
    with open(filename, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode()
    
def interactive_images(path, labels, images):
    """
    generates a interactive plot that spitches between images based on a dropdown menue 

    Inputs:
        path (str): path to where it saves the html
        lables (list of str): drop down labels
        images (list of images): the images to be ploted

    """
    fig = go.Figure()

    image_dict = {
        'sizex': 10,
        'sizey': 5,
        'x': -1,
        'y': 4,
        'xref': 'x',
        'yref': 'y',
        'opacity': 1.0,
        'layer': 'below',
    }
    
    labels.append("select videos here")
    nb_buttons= len(labels)


    # default
    fig.update_layout(
        images=[
            dict(
                **image_dict,
                source=images[0],  
            )
        ],             
    )

    dropdown_event = [
        dict(
            label=labels[i],
            method="relayout",
            args=[
                {
                    "images": [
                        dict(
                            **image_dict,
                            source = None if i == 12 else images[i],  
                        )
                    ],
                }
            ],
        )
        for i in range(nb_buttons)
    ]

    axis = dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_event,
                direction="down",
                showactive=True,
                x=0,
                y=1.2,
                xanchor="left",
                yanchor="top",
            ),
        ],
        width=800,
        height=500,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=axis,
        yaxis=axis,
    )
    fig.show()
    fig.write_html(path)

def plot_word_cloud(video_text, name, title):
    """
    Scores the subjectivity of video titles.

    Inputs:
        video_title (str): Title of the YouTube video.

    Output:
        float: The subjectivity score 0 for neutral 1 for highly subjective.
    """
    video_text= str.split(video_text.to_string(index=False))
    video_text= pd.Series(video_text)
    #filtering  
    stop_words = set(STOPWORDS)
    filtered = [token for token in video_text if token.lower() not in stop_words]
    filtered = pd.Series(filtered)
    filtered = filtered.str.replace('.','')
    filtered = filtered.str.lower()  #remove duplicates with different capitalisation
    filtered = filtered[filtered.str.len() > 1] # remove single letters
    comment_words = ''
    
    for val in filtered:
        val = str(val)
        if(str.isascii(val)): # remove all non english letters (wordcloud struggles with them)
            tokens = val.split()
            comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 1000, height = 500,
                    background_color ='white',
                    stopwords = stop_words,
                    min_font_size = 10).generate(comment_words)
    
    # plot the WordCloud image       
                    
    plt.figure(figsize = (10, 10), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad = 0)
    plt.savefig('img/wordclouds/' + name, format='png', bbox_inches='tight')
    plt.close()

from openai import OpenAI
def subjectivity_score(video_title):
    """
    Scores the subjectivity of video titles.

    Inputs:
        video_title (str): Title of the YouTube video.

    Output:
        float: The subjectivity score 0 for neutral 1 for highly subjective.
    """
    client = OpenAI(
    api_key=  '' #insert key here (it was removed for obvious reasons)
    )
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant who only focuses on language subjectivity."},
            {"role": "user", "content": f"""
            your task is to evaluate the subjectivity of news video titles and give each one a score from 0 neutral to 1 highly subjective. 
            The topic does not matter but the phrasing of the reporting. as an example "Switzerland obliterates all other countries in 
            quality of life" would be more subjective than "Switzerland exceeds other countries in quality of life". only return the score

            Title: "{video_title}"
            """}
    ],
    max_tokens = 10,
    temperature = 0.0,
    )

    response = completion.choices[0].message.content
    time.sleep(0.1)
    return response

def add_subjectivity(video_df):
    """
    Adds a subjectivity column to the data frame based on video titles.

    Inputs:
        video_df (polars_dataframe): Dataframe of video attributes.

    Output:
        video_df (polars_dataframe): Dataframe of video attributes with the new subjectivity column.
    """
    scores = [
        subjectivity_score(video["title"]) for video in video_df.iter_rows(named=True)
    ]
    return video_df.with_columns(pl.Series("subjectivity", scores))


