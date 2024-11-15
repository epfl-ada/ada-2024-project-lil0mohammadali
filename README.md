
# The Spread of News during Crisis
## Abstract
A lot of people use YouTube as their source for news. In this project we try to understand how information during crisis is spread on Youtube. For this we look at 12 events in 4 categories happening between 2017 and 2019 in the US, Europe and Asia. We analyse the spread of those events by looking at US _News Updates_ channels.

## Research Questions
- How do US News report on events happening inside the US compared to other events happening in Europe or Asia?
- How does the report of events differ based on "categories" of events?

## Proposed additional datasets
- We enriched our channels metadata set by adding the country of the channel using the [Youtube API](https://developers.google.com/youtube/v3).

## Methods
#### Filter the initial data: 
The Youniverse dataset is big in size. As we only focus on a fraction of the data we propose the following pipeline to get the data we need:

- Get the Channels of Interest (CoI): 
    - filter out all channels which do not belong to the category _"News & Politics"_. For this we use the `df_channels_en.tsv` dataframe.
    - We only focus on channels providing News Updates having a highly activity. We only keep the channels with an average activity above *56* (corresponds to 4 videos per day). For this we use timeseries data the `df_timeseries_en.tsv`. 
        - to ease the handling of the big dataframes we do an initial filtering of `yt_metadata.jsonl` with the CoI obtained so far.
    - even though the authors of the Youniverse dataset already filtered non-english speaking channels it turns out that there are still a important fraction of Hindi and other languages News channel. We filter thus the CoI obtained by the two previous points using a LLM to predict the language of the channel. For this we sample 5 video titles and description and filter out the channels which have at least one non-english predicted video. 

- Get the Videos of Intrest (VoI) 
    - Only keepp videos from the CoI in `yt_metadata.jsonl`.
    - For every event get the VoI by searching the title, description and the tags with specific keywords.

At this point we have reduced the size of all the datafiles (except the comments). 

### 

## Proposed timeline and organization within the team
#### Week 1 (26.10.-01.11.):
- Find out how to treat big dataframes (🐋Lisa)
- Filter channels by category (🐋Lisa)
- Find each three events in categories geopolitical, natural, economical, political (🦖Leonie🦝Samuel🦔Jad🐦Jeffrey)

#### Week 2 (02.11.-8.11.):
- Filter channels by activity (🐋Lisa)
- Filter non-english channels using LLM (🐦Jeffrey)
- Prepare list of keywords for selected events (🦔Jad)
- Prepare statistcal test pipeline (🦝Samuel)

#### Week 3 (9.11.-15.11.):
- ReadMe.md file (🐋Lisa)
- Analysis on video titles (🦖Leonie)
- Filter videos with keywords in title and description (🦔Jad)
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
- When we try to compare between different event types, the actual number of events per category which is feasible to analyse, is very small (three in our case). We are afraid that due to this small number, our analysis will be very sensitive to the events we chose and a general conclusion will likely be very biased. Would it be better if we only focus on one type of events (i.e. geopolitical) and analyse more events?

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



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
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

