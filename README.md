
# The Spread of News During Crisis
## Abstract
A lot of people use YouTube as their source of news. In this project, we try to look at the news landscape on YouTube and see how the news is reported and how the public engages with it. For this, we analyze big US News Channels to see how they report on different events in the realms of politics, geopolitical conflicts, economic crises, and natural disasters happening in the US, Europe, and Asia. We will analyze if there are any overarching trends or biases between these events, like reporting style or duration. Then we will look at the public response to these news videos to see if certain video features correlate with more engagement for a specific event. Based on that information we then want to formulate suggestions for what features could lead to a more effective video to start discussions or attract more viewership.

## Research Questions
- How do US News Channels report on different events?
- How does the public respond to different types of videos concerning specific events?
- How does one make an effective news video for certain types of events?

## Proposed additional datasets
- We enriched our channel metadata set by adding the country of the channel using the [Youtube API](https://developers.google.com/youtube/v3).

## Methods
#### Filter the initial data: 
The Youniverse dataset is big. As we only focus on a fraction of the data we propose the following pipeline to get the data we need:

- Get the Channels of Interest (CoI): 
    - filter out all channels that do not belong to the category _"News & Politics"_. For this, we use the `df_channels_en.tsv` dataframe.
    - We only focus on channels providing News Updates having a high activity. We only keep the channels with an average activity above *56* (corresponds to 4 videos per day for 2 weeks). For this, we use timeseries data `the df_timeseries_en.tsv`. 
        - to ease the handling of the big dataframes we do an initial filtering of `yt_metadata.jsonl` with the CoI obtained so far.
    - even though the authors of the Youniverse dataset already filtered non-english speaking channels it turns out that there is still an important fraction of Hindi and other language News channels. We thus further filter the CoI obtained by the two previous points using OPENAI's CHATGPT API to predict the language of the channel. For this, we sample 5 video titles and descriptions and pass them into a prompt asking the LLM to analyze the text to determine the channel's language. If any of the 5 videos are labeled non-English, the channel is removed from the dataset. 
    - we also obtain country information for a majority of the CoI. This was done with the YouTube Data API. Since the channel country data was fetched from today, we assume the country is the same as when the dataset was formed. Around 40-50 channels did not have their country of origin in the API data, so manual verification was performed, by visiting the channel page and/or other social media like X and Facebook.

- Get the Videos of Interest (VoI) 
    - Only keep videos from the CoI in `yt_metadata.jsonl`.
    - For every event get the VoI by searching the title, description, and tags with specific keywords.
 
- Filter out relevant comments using VoI     

At this point, we have reduced the size of all the datafiles (except the comments). 

## Proposed timeline and organization within the team
#### Week 1 (26.10.-01.11.):
- Find out how to treat big dataframes (🐋Lisa)
- Filter channels by category (🐋Lisa)
- Find three events in each category: geopolitical, natural, economical, political (🦖Leonie🦝Samuel🦔Jad🐦Jeffrey)

#### Week 2 (02.11.-8.11.):
- Filter channels by activity (🐋Lisa)
- Filter non-english channels using LLM (🐦Jeffrey)
- Prepare a list of keywords for selected events (🦔Jad)
- Prepare statistical test pipeline (🦝Samuel)
- Get country information from channels using Youtube Data API (🐦Jeffrey)
- Filter videos with keywords in the title and description (🦔Jad)

#### Week 3 (9.11.-15.11.):
- ReadMe.md file (🐋Lisa)
- Analysis of video titles (🦖Leonie)
- Filter videos with keywords in the title and description (🦔Jad)
- Filter the comments dataset on AWS (🐦Jeffrey)
- Correlation matrix over different values (🦝Samuel)

#### Week 4 (30.11.-06.12.):
- analysis on comments

#### Week 5 (7.11.-13.12.):
- create interactive plots
- start html website

#### Week 6 (14.12.-20.12.):
- conclusion of analysis



## Questions for TAs
- When we try to compare between different event types, the actual number of events per category which is feasible to analyse, is very small (three in our case). We are afraid that due to this small number, our analysis will be very sensitive to the events we chose and a general conclusion will likely be very biased. Would it be better if we only focus on one type of event (i.e. geopolitical) and analyse more events?

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.
The helper functions for filtering out country data and checking for english, are found under utils/general_utils.py 



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
|   |   ├── general_utils.py
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

