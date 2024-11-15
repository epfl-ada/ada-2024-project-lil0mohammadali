import polars as pl
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def bad_dislikes(filtered_df_metadata):
    counter = 0
    for i in range(len(filtered_df_metadata)):
        if not isinstance(filtered_df_metadata['dislike_count'][i] , float):
            counter += 1

    print("counter for non-float: ", counter)
    print("null elements: ",sum(filtered_df_metadata['dislike_count'].is_null()))
    print("nan elements: ",filtered_df_metadata['dislike_count'].is_nan().sum())
    print("indefinite elements: ",filtered_df_metadata['dislike_count'].is_infinite().sum())

def bad_likes(filtered_df_metadata):
    counter = 0
    for i in range(len(filtered_df_metadata)):
        if not isinstance(filtered_df_metadata['like_count'][i] , float):
            counter += 1

    print("counter for non-float: ", counter)
    print("null elements: ",sum(filtered_df_metadata['like_count'].is_null()))
    print("nan elements: ",filtered_df_metadata['like_count'].is_nan().sum())
    print("indefinite elements: ",filtered_df_metadata['like_count'].is_infinite().sum())

def summarize_outliers(filtered_df_metadata):
    # If parsing fails (wrong format), `parsed_date` will contain `null`
    filtered_df_metadata = filtered_df_metadata.with_columns(
        pl.col("upload_date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("parsed_date")
    )

    # Count rows where "upload_date" does not match the format (i.e., where "parsed_date" is null)
    date_outlier_count = filtered_df_metadata.filter(pl.col("parsed_date").is_null()).height

    # Count rows where "like_count" is null directly
    like_outlier_count = filtered_df_metadata.select(pl.col("like_count").is_null().sum()).item()

    print(f"Total number of videos: {filtered_df_metadata.shape[0]:,}")
    print(f"Date Outlier Count: {date_outlier_count:,}")
    print(f"Like/Dislike Outlier Count (null elements): {like_outlier_count:,}")   


def filtering_bad_rows(filtered_df_metadata):
    # filtering out all the bad rows
    print(f"Original metadata shape: ({filtered_df_metadata.shape[0]:,}, {filtered_df_metadata.shape[1]:,})")

    # type 1: like and dislike have None entry

    # Attempt to parse the date strings; rows that don't match will be set to `None`
    filtered_df_metadata = filtered_df_metadata.with_columns(
        pl.col("upload_date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("parsed_date")
        # pl.col("upload_date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).dt.strftime("%Y-%m-%d").str.strptime(pl.Date, "%Y-%m-%d").alias("upload_date")
        # pl.col("upload_date").dt.strftime("%Y-%m-%d").str.strptime(pl.Date, "%Y-%m-%d").alias("upload_date")
    )

    # Filter out rows where parsing failed (i.e., where "parsed_date" is None)
    filtered_df_metadata = filtered_df_metadata.filter(pl.col("parsed_date").is_not_null())

    # type 2: upload date have non-parseable entries

    filtered_df_metadata = filtered_df_metadata.filter(
        pl.col("like_count").is_not_null()
    )

    filtered_df_metadata.drop_in_place('parsed_date')

    print(f"New metadata shape: ({filtered_df_metadata.shape[0]:,}, {filtered_df_metadata.shape[1]:,})")

    return filtered_df_metadata


def remove_hour(filtered_df_metadata):
    return filtered_df_metadata.with_columns(pl.col("upload_date").str.strptime(pl.Date,"%Y-%m-%d %H:%M:%S").dt.strftime("%Y-%m-%d").str.strptime(pl.Date, "%Y-%m-%d").alias("upload_date"))


##### section 1: political events ######
# 2016 US elections
list1_1 = ["Clinton", "Hillary", "Trump", "Sanders", "Pence", "Kaine", "leak", "swing state", "swing states", 
          "GOP", "DNC", "RNC", "primary", "electoral", "campaign", "republican", "democrat", "potus", "ballot"]
# 2019 indian elections
list1_2 = ["Modi", "Narendra", "BJP", "Congress", "Rahul Gandhi", "Lok Sabha", "manifesto", 
          "chowkidar", "Vande Mataram", "NYAY", "Bharatiya Janata", "NDA", "UPA"]                       ### might need cleaning up
# 2019 EU elections
list1_3 = ["European Parliament", "EU election", "Eurosceptic", "EU27", "European Greens", "pro-EU", "anti-EU", "european election"]

##### section 2: economic crises ######
# Venezuela Hyperinflation in 2018
list2_1 = ["Venezuela", "hyperinflation", "bolivar", "Maduro", "inflation crisis", "black market", "currency collapse", "economic crisis", "food shortage", "Venezuelan central bank"]
# US-China trade war 2018
list2_2 = ["US-China", "trade war", "tariffs", "Trump", "Beijing", "import duties", "trade deficit", "Chinese goods", "trade negotiations", "intellectual property"]
# Greece Economic Crisis
list2_3 = ["Greece", "Greek debt", "austerity", "bailout", "Eurozone crisis", "Troika", "IMF", "Greek banks", "sovereign debt", "economic reforms"]

###### section 3: natural disasters ######
# Hurricane Harvey (2017) - US
list3_1 = [
    "Hurricane Harvey", "Harvey storm", "Harvey flood", "Houston flooding", 
    "Texas hurricane", "Houston disaster", "Harvey relief", "Harvey rescue", 
    "Harvey aftermath", "Harvey damage", "FEMA Harvey", "Harvey victims", 
    "Harvey recovery", "Harvey Texas", "Harvey impact", "Harvey landfall",
    "Harvey evacuation", "Hurricane 2017"]
# Sulawesi Earthquake and Tsunami (2018) - Asia
list3_2 = [
    "Sulawesi earthquake", "Indonesia earthquake", "Palu tsunami", 
    "Sulawesi disaster", "Indonesia tsunami", "Sulawesi quake", 
    "Palu earthquake", "Indonesian relief", "Sulawesi aid", "Sulawesi 2018", 
    "Sulawesi rescue", "Palu crisis", "Sulawesi recovery"]
# European Heatwaves (2019) - Europe
list3_3 = [
    "heatwave 2019", "2019 heatwave", "France heatwave", 
    "Germany heatwave", "UK heatwave", "europe heatwave", "heatwave"]

###### section 4: geopolitical conflicts ######
# 2017 Raqqa battle
list4_1 = [
    "syria", "raqqa"]
# Asia: India-Pakistan Conflict (Pulwama and Balakot Airstrikes)
list4_2 = [
    "pulwama", "balakot", "abhinandan"]                                                                               # cashmir conflict, possibly masood azhar
# "Europe: 2015-2017 Rise in ISIS Attacks"
list4_3 = [
    "paris attack", "brussels bombing", 
    "bataclan", "nice truck", "charlie hebdo", "manchester bombing"]                                           ##### test difference with attack vs attacks
   # would having different events be a good idea?? maybe stick to one

titles = [
    "2016 US elections",
    "2019 indian elections",
    "2019 EU elections",
    
    "Venezuela Hyperinflation in 2018",
    "US-China trade war 2018",
    "Greece Economic Crisis 2015",
    
    "Hurricane Harvey (2017)",
    "Sulawesi Earthquake and Tsunami (2018)",
    "European Heatwaves (2019)",
    
    "2017 battle of Raqqa (by US-led coalition)",
    "2019 India-Pakistan Conflict (Pulwama and Balakot Airstrikes)",
    "2015-2017 Rise in ISIS Attacks in Europe"
]
list_of_lists = [list1_1, list1_2, list1_3, list2_1, list2_2, list2_3, list3_1, list3_2, list3_3, list4_1, list4_2, list4_3]
# index :           0        1        2        3        4        5        6        7        8        9       10       11


def plot_update_frequ(index, filtered_df_metadata, all_plots = False, grouping_mode = "daily"):
    
    if not all_plots:

        terms = list_of_lists[index]
        title = titles[index]
        print(f"Event: {title}")

        pattern = "(?i)" + "|".join([f"{term}" for term in terms])       # accept words where the term appears as part of the word, so Leak --> Leaked would be counted. Also case insensitive

        event_metadata = filtered_df_metadata.filter(
            (pl.col("description").str.contains(pattern, literal=False)) |
            (pl.col("title").str.contains(pattern, literal=False))
        )
        print(f"Related videos found: {event_metadata.shape[0]:,}")

        match grouping_mode:
            case "daily":
                date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
            case "weekly":
                date_counts = event_metadata.with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date")).group_by("upload_date").count().sort("upload_date")
            case "monthly":
                date_counts = event_metadata.with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date")).group_by("upload_date").count().sort("upload_date")

        # Plot the histogram
        plt.figure(figsize=(16, 6))
        plt.bar(date_counts["upload_date"], date_counts["count"], width = 3)    # adjust width
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title(f"Upload frequency for videos related to {title}")
        dates = date_counts["upload_date"].to_list()

        # Set x-axis ticks to show every month
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Change `interval` to control frequency
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format as "Year-Month"

        # # Set x-axis ticks to show every month
        # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Change `interval` to control frequency
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format as "Year-Month"

        # # Set x-axis ticks to show every two weeks
        # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Every two weeks
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format as "Year-Month-Day"

        plt.xticks(rotation=90)

        start_date = datetime(2015, 1, 1)
        end_date = datetime(2020, 1, 1)
        plt.xlim(start_date, end_date)
        
        plt.grid()
        plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(12, 1, figsize=(16, 36), constrained_layout=True)
        for index in range(12):
                
            terms = list_of_lists[index]
            title = titles[index]
            print(f"Event: {title}")

            pattern = "(?i)" + "|".join([f"{term}" for term in terms])       # accept words where the term appears as part of the word, so Leak --> Leaked would be counted. Also case insensitive

            event_metadata = filtered_df_metadata.filter(
                (pl.col("description").str.contains(pattern, literal=False)) |
                (pl.col("title").str.contains(pattern, literal=False))
            )
            print(f"Related videos found: {event_metadata.shape[0]:,}")
            print("------------")

            match grouping_mode:
                case "daily":
                    date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
                case "weekly":
                    date_counts = event_metadata.with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date")).group_by("upload_date").count().sort("upload_date")
                case "monthly":
                    date_counts = event_metadata.with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date")).group_by("upload_date").count().sort("upload_date")

            
            # Plot the histogram
            ax[index].bar(date_counts["upload_date"], date_counts["count"], width = 3)    # adjust width
            ax[index].set_xlabel("Date")
            ax[index].set_ylabel("Count")
            ax[index].set_title(f"Upload frequency for videos related to {title}")
            dates = date_counts["upload_date"].to_list()

            # Set x-axis ticks to show every month
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Change `interval` to control frequency
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format as "Year-Month"

            # # Set x-axis ticks to show every month
            # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Change `interval` to control frequency
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format as "Year-Month"

            # # Set x-axis ticks to show every two weeks
            # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Every two weeks
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format as "Year-Month-Day"

            plt.xticks(rotation=90)

            start_date = datetime(2013, 1, 1)
            end_date = datetime(2019, 12, 1)
            ax[index].set_xlim(start_date, end_date)
            
            ax[index].grid(True)
        plt.tight_layout()
        plt.show()

