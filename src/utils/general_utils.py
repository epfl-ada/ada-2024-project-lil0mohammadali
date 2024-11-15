import keys
from googleapiclient.discovery import build
from openai import OpenAI
import time

YOUTUBE_KEY = keys.YOUTUBE_API_KEY
OPEN_API_KEY = keys.OPENAI_API_KEY

# def write_polars_to_csv(polars_dataframe, name):
# # # Convert to pandas DataFrame
#     polars_dataframe = polars_dataframe.to_pandas()

#     # Write the DataFrame to a CSV file
#     polars_dataframe.to_csv(f"{name}.csv", index=False)

youtube_api = build('youtube', 'v3', developerKey=YOUTUBE_KEY)
def get_channel_country(youtube_api, channel_id):
    request = youtube_api.channels().list(
        part="snippet",
        id=channel_id
    )
    
    response = request.execute()
    
    # Check if the response contains the necessary information
    if "items" in response and len(response["items"]) > 0:
        country = response["items"][0]["snippet"].get("country", "Country not available")
        return country
    else:
        return "Channel not found"

client = OpenAI(api_key=OPEN_API_KEY)

# Define the function to detect the language using ChatGPT
def check_video_language(video_title, video_description, closed_captions=""):
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

def check_channel_english(final_df, channel_id):
    videos = final_df.loc[final_df['channel_id'] == channel_id]
    for index, video in videos.iterrows():
        # Check if the text is in English using CHATGPT API
        is_english = check_video_language(video_title=video['title'], video_description=video['description'])
        if not is_english:
            print("channel is not english")
            return False  # If any video is not English, return False
        time.sleep(0.5)
    print("channel is english")
    return True  # If all videos checked are English, return True