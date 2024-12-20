
# The Spread of News During Crisis
## Abstract
A lot of people use YouTube as their source of news. In this project, we try to look at the news landscape on YouTube and see how the news is reported and how the public engages with it. For this, we analyze big US News Channels to see how they report on different events in the realms of geopolitical conflicts, and natural disasters happening in the US, Europe, and Asia. We analyzeed if there are any overarching reporting trends and found that the reporting is signifcantly skewd towards natural disasters in the US. Out of the 11 thousand videos that were analyzed 67% if them where about natural disasters in the US. We also looked at the public response to these news videos to see if certain video features correlate with more engagement for a specific event and found ????????. Based on that information we suggest that ?????????? could lead to a more effective video to start discussions or attract more viewership.

## Research Questions
- How does the reporting of events by US news channels change with respect to the type of event as well as its location?
- How is the public's response to an event affected by its nature, location, and video format through which it is presented?
- How does one make an effective news video to ellicit specific reactions and levels of interaction from the public?


## Methods
#### Filter the initial data: 
The Youniverse dataset is big. As we only focus on a fraction of the data we used the following pipeline to get the data we need:
Below is a figure describing at a high level the filtering of the channels with a description below
 ![channel_filter_pipeline](img/channel_filter_pipeline.jpg)

- **Get the Channels of Interest (CoI):** 
    - Filter out all channels that do not belong to the category "News & Politics". For this, we use the df_channels_en.tsv dataframe.
    - We only focus on channels providing News Updates having a high activity. We only keep the channels with an average activity above  to 4 videos per day for 2 weeks. For this, we use timeseries data the df_timeseries_en.tsv. 
    - Even though the Youniverse dataset is supposed to only contain English speaking channels, especially Hindi and Arabic speaking channels were still present. We thus further filter the CoI obtained by the two previous steps using OpenAI's ChatGPT API to predict the language of the channel. For this, we sample 5 video titles and descriptions and pass them into a prompt asking the LLM to analyze the text to determine the channel's language. If any of the 5 videos is labeled non-English, the channel is removed from the dataset. 
    - We also obtain country information for a majority of the CoI. This was done with the YouTube Data API. Since the channel country data was fetched from today, we assume the country is the same as when the dataset was formed.

- **Get the Videos of Interest (VoI)**  
    In order to get the VoI we complied a list of keywords conserning each event. 
    - The choice of events was rigorous, as we only wanted events that had a clear starting date that accurately represents the response to breaking news and that we can use for filtration. We also looked for the most popular and impactful events for each category, i.e. the ones that garnered the most reporting, to maximize the datapoints and best represent the event types. 
    - While comilping the list of keywords, the resulting videos were continously sampled to monitor not only the amount of videos that were returned but also to make sure that there was no significant amount of videos returned that did not concern the event. 
    - The VoI contain at least one very specific key word or a combination of multiple less specific keywords.    

- **Filter out relevant comments using VoI**   
    Once we had the VoI we could filter the comments datasets using the video ID to obtain the metrics of number of comments and number of comment-replies for each VoI.

## Analysis

The analysis is decomposed into two sections: 1. How does US News report on different events based on category and location? (i.e. studying the reporting side) 2. How does the public respond to events and specific video formats? 

### Video metrics
To answer the first question, we look at videos related to each event, getting statistics for: 

    - video duration
    - type of video (live footage)
    - capitalization ratio of title
    - frequency of video uploads at the time upload
    - appearance of specific keywords in the title: “breaking” and “update”
    - subjectivity score of the title


The video duration was taken directly from the YouNiverse dataset. The type of video reflects if it shows ground footage or not, and the filtering is done based on wether or not the word “footage” appears in the title. The frequeny of video uploads describles the average daily upload frequency of the specific channel in the 2 weeks surrounding the upload of that specific video. The video title are offer us a few metrics. The caplitalization of the title is the ratio of upper case letters in the title. The title is also used to sarch for common keywords, mainly “breaking” and “update”. Finally we generate a subjectivity score for each title using OpenAI’s API using the following promt:

“your task is to evaluate the subjectivity of news video titles and give each one a score from 0 neutral to 1 highly subjective. The topic does not matter but the phrasing of the reporting. As an example “Switzerland obliterates all other countries in quality of life” would be more subjective than “Switzerland exceeds other countries in quality of life”. Only return the score”

We can see in our analysis that for the US and Asia, subjectivity seems to be on average significantly higher for geopolitical events, whereas we don’t see a massive difference in European events. Video duration seems to be longer for geopolitical events accross the board, potentially reflecting that such events need more context and insights in order to understand the situation. Finally, in geopolitics, US-related events present higher subjectivity than Europe and Asia, as for environmental ones, Europe presents the highest subjectivity in reporting.

### Public response
To answer the second question, we look at data relating to the public response to videos.  To do so, and after considering numerous options, we landed on the following four:

    - views
    - (likes - dislikes) / views
    - number of comments / views
    - average replies per comment

The choice of the first metric is quite straightforward. For the second metric we tried to find a measure for the general desire of the user to see more of the same or similar content, and a dislike as a desire not to. Taking the difference gives us the net general interest of the users in this content. Given that likes and dislikes grow with views, we normalize this difference by view count. For the same reason, we normalize the number of comments by views to obtain the third metric. This enables us to assess the extent to which a particular video entices people to express their views in the videos’ content in the comments. And the average replies per comment is a metric reflecting how much the video encourages discussion.

????????????? replace everything after this

We would like to find correlations between the statistic of the first question and those of the public's response (using t and f tests to see the significance of correlation), potentially finding meaningful patterns. The goal of this is provide tips for useful features that news companies and NGO's can use to better engage users. 

Based on the metrics of the public's response we could classify the reaction into two main categories: relatively high view count and low comments/replies to comments, and average views with high comments/replies to comments. The first type would reflect virality and high reach of the video, while the second indicates that the video prompts strong user ungagement, encouraging discussions within the public. We could thereafter determine what format of videos result in high virality vs discussions, and news channels could adapt their videos according to the desired outcome. Like/dislike ratio could also be used to try determining how to potentially minimize division among the public (indicated by a ratio close to 1). 

## Timeline and organization within the team
#### Week 1 (26.10.-01.11.):
- Find out how to treat big dataframes (🐋Lisa)
- Filter channels by category (🐋Lisa)
- Find three events in each category: geopolitical, natural, economical, political (🦖Leonie🦝Samuel🦔Jad🐦Jeffrey)

#### Week 2 (02.11.-8.11.):
- Filter channels by activity (🐋Lisa)
- Filter non-english channels using LLM (🐦Jeffrey)
- Prepare a list of keywords that would isolate specific events and filter out the related videos by searching in title and description (🦔Jad)
- Prepare statistical test pipeline (🦝Samuel)
- Get country information from channels using Youtube Data API (🐦Jeffrey)

#### Week 3 (9.11.-15.11.):
- ReadMe.md file (🐋Lisa)
- Analysis of video titles (🦖Leonie)
- Study of the upload frequency evolution for each event(🦔Jad)
- Filter the comments dataset on AWS (🐦Jeffrey)
- Correlation matrix over different values (🦝Samuel)

#### Week 4 (30.11.-06.12.):
- Filtering for events (🦔Jad)
- Improving correlation matrix over different values (🦝Samuel)
- Filter the comments dataset on AWS (🐦Jeffrey)
- make the website template (🦖Leonie)

#### Week 5 (7.11.-13.12.):
- create interactive plots (🐦Jeffrey)
- creat interactive plots (🐋Lisa)
- subjectivity score using LLM (🦖Leonie)
- classifying videos (🦔Jad)
- statistical analysis (🦝Samuel)


#### Week 6 (14.12.-20.12.):
- visualisation (🐦Jeffrey & 🦝Samuel)
- website (🦖Leonie)
- jypiter notebook (🐋Lisa)
- linear regression model (🦔Jad)
- writing (🦔Jad)





