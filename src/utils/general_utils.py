from . import keys
from googleapiclient.discovery import build
from openai import OpenAI
import time

YOUTUBE_KEY = keys.YOUTUBE_API_KEY
OPEN_API_KEY = keys.OPENAI_API_KEY

youtube_api = build('youtube', 'v3', developerKey=YOUTUBE_KEY)

def get_channel_country(channel_id):
    """
        Fetch the country of a YouTube channel.

        Input: 
            channel_id (str): The unique ID of the channel.
        Output: 
            str: The country code, "Country not available" if missing, 
            or "Channel not found".
    """
    request = youtube_api.channels().list(
        part="snippet",
        id=channel_id
    )
    
    response = request.execute()
    
    # check if the response contains the necessary information
    if "items" in response and len(response["items"]) > 0:
        country = response["items"][0]["snippet"].get("country", "Country not available")
        return country
    else:
        return "Channel not found"

client = OpenAI(api_key=OPEN_API_KEY)


def check_video_language(video_title, video_description, closed_captions=""):
    """
    Checks if the title and description of a YouTube video are in English.

    Input: 
        video_title (str): Title of the YouTube video.
        video_description (str): Description of the YouTube video.
        closed_captions (str, optional): Closed-Captions text.

    Output: 
        bool: True if the text is English else False.
    """
    # print("title: ", video_title)
    # print("description: ", video_description)

    messages = [
        {"role": "system", "content": "You are a helpful assistant who only focuses on language identification."},
        {"role": "user", "content": f"""
        Given the title and description of a YouTube video, please determine if the text is in English. Ignore URLs and non-English symbols. 
        Respond with "yes" if you think the text is in English, and "no" if you think it is not.

        Title: "{video_title}"
        Description: "{video_description}"

        Is the text in English?
        """}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=messages
    )

    response = completion.choices[0].message.content

    # print("Chat response: ", response)
    # Check if the response includes "yes"
    return "yes" in response

def check_channel_english(videos_df, channel_id):
    """
    Checks if all videos in a channel have English titles and descriptions.

    Inputs:
        videos_df (pandas.DataFrame): DataFrame containing videos from the desired channels
        channel_id (str): The ID of the YouTube channel to check.

    Output:
        bool: True if all videos of the channel have titles and descriptions in English, else False.
    """
    videos = videos_df.loc[videos_df['channel_id'] == channel_id]
    for index, video in videos.iterrows():
        # check if the text is in English using CHATGPT API
        is_english = check_video_language(video_title=video['title'], video_description=video['description'])
        if not is_english:
            print("channel is not english")
            return False  # if any video is not English, return False
        time.sleep(0.5)
    print("channel is english")
    return True  # if all videos checked are English, return True