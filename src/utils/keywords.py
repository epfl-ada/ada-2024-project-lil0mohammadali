# this file contains the updated filtering function with only natural disasters and geopolitical conflicts

import polars as pl
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
import numpy as np

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


hurricane_harvey_keywords = [           # ok
    ["Hurricane", "Harvey"], 
    ["Texas", "flood"], 
    ["Houston", "flood"], 
    ["Harvey", "storm"], 
    ["Harvey", "disaster"], 
    ["Houston", "hurricane"], 
    ["Harvey", "relief", "efforts"],
    ["harvey", "houston"],
]

california_wildfires_keywords = [       # ok
    ["California", "wildfire"], 
    ["California", "firestorm"], 
    ["California", "fire"],

    ["Camp", "Fire"], 
    ["Paradise", "fire"],     
    ["Camp", "wildfire"], 
    ["Paradise", "wildfire"],  
]

hurricane_maria_keywords = [            # ok
    ["Hurricane", "Maria"], 
    ["Puerto", "Rico", "maria"], 
    ["Maria", "disaster"], 
    ["Maria", "flood"], 
    ["Maria", "aftermath"]
]

hurricane_michael_keywords = [          # ok
    ["michael", "hurricane"], 
]

sulawesi_earthquake_tsunami_keywords = [    # ok
    ["Sulawesi", "earthquake"], 
    ["Sulawesi", "tsunami"], 
    ["Indonesia", "earthquake"], 
    ["Indonesia", "tsunami"], 
]

nepal_earthquake_keywords = [               # ok
    ["Kathmandu", "quake"], 
    ["Nepal", "quake"]
]

bangladesh_cyclone_mora_keywords = [        # ok
    ["Cyclon", "Mora"],
]

india_floods_keywords = [                   # ok (but check if want to include more floods)
    # ["India", "flood"],    
    ["Kerala", "flood"]
]

heatwaves_2019_keywords = [             # ok
    ["Europe", "heatwave"], 
    ["Belgium", "heatwave"],
    ["Germany", "heatwave"],
    ["Luxembourg", "heatwave"],
    ["Netherlands", "heatwave"],
    ["UK", "heatwave"],
    ["United Kingdom", "heatwave"],
    ["France", "heatwave"],

    ["Europe", "heat wave"], 
    ["Belgium", "heat wave"],
    ["Germany", "heat wave"],
    ["Luxembourg", "heat wave"],
    ["Netherlands", "heat wave"],
    ["UK", "heat wave"],
    ["United Kingdom", "heat wave"],
    ["France", "heat wave"],
]

portugal_wildfires_keywords = [         # ok
    ["Portugal", "fire"], 
    ["Pedrógão", "fire"], 
]

european_floods_2014_keywords = [       # ok
    ["Europe", "flood"], 
    ["Balkan", "flood"], 
    ["Bosnia", "flood"], 
    ["Serbia", "flood"], 
    ["Croatia", "flood"], 
    ["Romania", "flood"], 
    ["Bulgaria", "flood"]
]

greek_wildfires_keywords = [            # ok
    ["Athens", "fire"], 
    ["Greece", "fire"],
    ["Greek", "fire"]
]


italy_earthquakes_keywords = [          # ok
    ["Amatrice", "quake"], 
    ["italy", "quake"],
    ["italian", "quake"],
]

# Keyword lists for geopolitical and armed conflicts

# US Involvement Conflicts
mosul_offensive_keywords = [            # ok
    ["Mosul", "Offensive"], 
    ["Mosul", "battle"], 
    ["Mosul", "US"], 
    ["Mosul", "strike"], 
    ["Mosul", "fight"], 
    ["Mosul", "combat"], 
    ["Mosul", "capture"],
    ["Mosul", "attack"]
]

battle_of_kobani_keywords = [           # ok
    ["kobani"]
]

battle_of_raqqa_keywords = [            # ok
    ["raqqa"]
]

kunduz_city_attack_keywords = [         # ok
    ['kunduz']
]

battle_of_sirte_keywords = [            # ok
    ["Sirte"]
]

# Asia Conflicts
india_pakistan_conflict_keywords = [     # ok
    ["Pulwama"],
    ["Balakot"],
]

syrian_civil_war_keywords = [               # ok
    ["Aleppo"]
]

yemeni_civil_war_keywords = [               # ok
    ['yemen'],
    ['hudaida'],
    ['hudaydah']
]

nagorno_karabakh_conflicts_keywords = [     # ok
    ['nagorno'],
    ["karabakh"],
    ["armenia", "azerbaijan"]
]

# Europe Conflicts
crimea_annexation_keywords = [              # ok
    ["Crimea", "Annex"], 
    ["Russia", "Ukraine"], 
    ["Eastern", "Ukraine"], 
    ["Donbas"], 
    ["separatist", "ukrain"],
    ["separatist", "russia"]
]

list_of_lists_environment = [
    # Natural Disasters
    hurricane_harvey_keywords, 
    california_wildfires_keywords, 
    hurricane_maria_keywords, 
    hurricane_michael_keywords,

    sulawesi_earthquake_tsunami_keywords, 
    nepal_earthquake_keywords,
    bangladesh_cyclone_mora_keywords, 
    india_floods_keywords, 

    heatwaves_2019_keywords, 
    portugal_wildfires_keywords, 
    european_floods_2014_keywords, 
    greek_wildfires_keywords, 
    italy_earthquakes_keywords
]

list_of_lists_conflict = [
    # Geopolitical and Armed Conflicts (US Involvement)
    mosul_offensive_keywords, 
    battle_of_kobani_keywords, 
    battle_of_raqqa_keywords, 
    kunduz_city_attack_keywords, 
    battle_of_sirte_keywords,

    # Geopolitical and Armed Conflicts (Asia)
    india_pakistan_conflict_keywords, 
    syrian_civil_war_keywords, 
    yemeni_civil_war_keywords, 

    # Geopolitical and Armed Conflicts (Europe)
    crimea_annexation_keywords, 
    nagorno_karabakh_conflicts_keywords, 

]

titles_environment = [
    # US
    "Hurricane Harvey (2017)", 
    "California Wildfires (2018)", 
    "Hurricane Maria (2017)", 
    "Hurricane Michael (2018)", 

    # Asia
    "Sulawesi Earthquake and Tsunami (2018)", 
    "Nepal Earthquake (2015)", 
    "Bangladesh Cyclone Mora (2017)", 
    "India Floods (2018)", 

    # Europe
    "Europe Heatwaves (2019)", 
    "Portugal Wildfires (2017)", 
    "European Floods (2014)", 
    "Greek Wildfires (2018)", 
    "Italy Earthquakes (2016)"
]

titles_conflict = [

    # Geopolitical Conflicts / Armed Conflicts (US)
    "Mosul Offensive (2016-2017) - Iraq",
    "Battle of Kobani (2014-2015) - Syria",
    "Battle of Raqqa (June - October 2017) - Syria",
    "Kunduz City Attack (2015) - Afghanistan",
    "Battle of Sirte (2016) - Libya",
    
    # Geopolitical Conflicts / Armed Conflicts (Asia)
    "Pulwama attack and Balakot Airstrikes, kashmir conflict (February 2019) - India / Pakistan",
    "Syrian Civil War, Aleppo Offensive (2016))",
    "Yemeni Civil War , battle of Hudayah (June - December 2018 )",
    
    # Geopolitical Conflicts / Armed Conflicts (Europe)
    "Crimea Annexation and Conflict in Eastern Ukraine (2014)",
    "Nagorno-Karabakh Conflicts (Clashes (2016)) - Armenia-Azerbaijan",
]


def title_or_desc_contains(term):
    pattern = f"(?i){term}"  # case-insensitive
    return (pl.col("title").str.contains(pattern, literal=False)) | (pl.col("description").str.contains(pattern, literal=False))

def build_filter_condition(terms):
    # Ensure terms is a list of lists for uniform processing
    # If we detect terms is a flat list, we wrap each element into its own list.
    complex_terms = []
    for t in terms:
        if isinstance(t, str):
            complex_terms.append([t])
        else:
            # We assume it's already a list of strings
            complex_terms.append(t)
    
    # Now complex_terms is a list of lists, each sub-list is an AND group
    final_condition = pl.lit(False)
    for group in complex_terms:
        group_condition = pl.lit(True)
        for term in group:
            group_condition = group_condition & title_or_desc_contains(term)
        final_condition = final_condition | group_condition
    
    return final_condition


def plot_update_freq_v2(index, filtered_df_metadata, all_plots=False, grouping_mode="daily", event_type = None):

    if not all_plots:
        # Retrieve terms and title for the given index
        list_of_lists = list_of_lists_environment + list_of_lists_conflict
        titles = titles_environment + titles_conflict

        terms = list_of_lists[index]
        title = titles[index]
        
        # Build the filtering condition based on complex terms logic
        final_condition = build_filter_condition(terms)
        print(f"Event: {title}")

        event_metadata = filtered_df_metadata.filter(final_condition)
        print(f"Related videos found: {event_metadata.shape[0]:,}")

        match grouping_mode:
            case "daily":
                date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
            case "weekly":
                date_counts = (event_metadata
                               .with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date"))
                               .group_by("upload_date")
                               .count()
                               .sort("upload_date"))
            case "monthly":
                date_counts = (event_metadata
                               .with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date"))
                               .group_by("upload_date")
                               .count()
                               .sort("upload_date"))

        # Plot the histogram
        plt.figure(figsize=(16, 6))
        plt.bar(date_counts["upload_date"], date_counts["count"], width=3)  # adjust width
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title(f"Upload frequency for videos related to {title}")

        start_date = datetime(2018, 1, 1)
        end_date = datetime(2018, 12, 1)
        # plt.xlim(start_date, end_date)

        # Set x-axis ticks to show yearly
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.show()

    else:
        if event_type == "environment":
            list_of_lists = list_of_lists_environment
            titles = titles_environment
        elif event_type == "conflict":
            list_of_lists = list_of_lists_conflict
            titles = titles_conflict
        else:
            raise KeyError("wrong event type: \"environment\" or \"conflict\" ")

        # Plot all events in a single figure
        fig, ax = plt.subplots(len(list_of_lists), 1, figsize=(16, 36), constrained_layout=True)
        for idx in range(len(list_of_lists)):
            
            terms = list_of_lists[idx]
            title = titles[idx]

            print(f"Event: {title}")

            final_condition = build_filter_condition(terms)
            event_metadata = filtered_df_metadata.filter(final_condition)
            print(f"Related videos found: {event_metadata.shape[0]:,}")
            print("------------")

            match grouping_mode:
                case "daily":
                    date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
                case "weekly":
                    date_counts = (event_metadata
                                   .with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date"))
                                   .group_by("upload_date")
                                   .count()
                                   .sort("upload_date"))
                case "monthly":
                    date_counts = (event_metadata
                                   .with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date"))
                                   .group_by("upload_date")
                                   .count()
                                   .sort("upload_date"))

            ax[idx].bar(date_counts["upload_date"], date_counts["count"], width=3)  # adjust width
            ax[idx].set_xlabel("Date")
            ax[idx].set_ylabel("Count")
            ax[idx].set_title(f"Upload frequency for videos related to {title}")

            # Use yearly ticks
            ax[idx].xaxis.set_major_locator(mdates.YearLocator())
            ax[idx].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax[idx].xaxis.get_majorticklabels(), rotation=90)

            start_date = datetime(2013, 1, 1)
            end_date = datetime(2019, 12, 1)
            ax[idx].set_xlim(start_date, end_date)
            
            ax[idx].grid(True)

        plt.show()

def plot_update_freq_v3(index, filtered_df_metadata, all_plots=False, grouping_mode="daily", event_type=None, start_date=None): # added argument start_date

    if not all_plots:
        # Retrieve terms and title for the given index
        list_of_lists = list_of_lists_environment + list_of_lists_conflict
        titles = titles_environment + titles_conflict

        terms = list_of_lists[index]
        title = titles[index]
        
        # Build the filtering condition based on complex terms logic
        final_condition = build_filter_condition(terms)
        print(f"Event: {title}")

        event_metadata = filtered_df_metadata.filter(final_condition)
        
        # Apply start_date filter if provided
        if start_date:
            event_metadata = event_metadata.filter(pl.col("upload_date") > start_date)

        print(f"Related videos found: {event_metadata.shape[0]:,}")

        match grouping_mode:
            case "daily":
                date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
            case "weekly":
                date_counts = (event_metadata
                               .with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date"))
                               .group_by("upload_date")
                               .count()
                               .sort("upload_date"))
            case "monthly":
                date_counts = (event_metadata
                               .with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date"))
                               .group_by("upload_date")
                               .count()
                               .sort("upload_date"))

        # Plot the histogram
        plt.figure(figsize=(16, 6))
        plt.bar(date_counts["upload_date"], date_counts["count"], width=3)
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title(f"Upload frequency for videos related to {title}")

        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.xlim(datetime(2013, 1, 1), datetime(2019, 12, 1))
        plt.show()

    else:
        if event_type == "environment":
            list_of_lists = list_of_lists_environment
            titles = titles_environment
        elif event_type == "conflict":
            list_of_lists = list_of_lists_conflict
            titles = titles_conflict
        else:
            raise KeyError("wrong event type: \"environment\" or \"conflict\" ")

        fig, ax = plt.subplots(len(list_of_lists), 1, figsize=(16, 36), constrained_layout=True)
        for idx in range(len(list_of_lists)):
            terms = list_of_lists[idx]
            title = titles[idx]

            print(f"Event: {title}")

            final_condition = build_filter_condition(terms)
            event_metadata = filtered_df_metadata.filter(final_condition)
            
            if start_date:
                event_metadata = event_metadata.filter(pl.col("upload_date") > start_date)

            print(f"Related videos found: {event_metadata.shape[0]:,}")
            print("------------")

            match grouping_mode:
                case "daily":
                    date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
                case "weekly":
                    date_counts = (event_metadata
                                   .with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date"))
                                   .group_by("upload_date")
                                   .count()
                                   .sort("upload_date"))
                case "monthly":
                    date_counts = (event_metadata
                                   .with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date"))
                                   .group_by("upload_date")
                                   .count()
                                   .sort("upload_date"))

            ax[idx].bar(date_counts["upload_date"], date_counts["count"], width=3)
            ax[idx].set_xlabel("Date")
            ax[idx].set_ylabel("Count")
            ax[idx].set_title(f"Upload frequency for videos related to {title}")

            ax[idx].xaxis.set_major_locator(mdates.YearLocator())
            ax[idx].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax[idx].xaxis.get_majorticklabels(), rotation=90)
            ax[idx].set_xlim(datetime(2013, 1, 1), datetime(2019, 12, 1))
            ax[idx].grid(True)

        plt.show()

def plot_update_freq_v4(index, filtered_df_metadata, all_plots=False, grouping_mode="daily", event_type=None, crop_time=False): # added argument crop_time

    event_dates = [
        [date(2017, 8, 1), date(2018, 7, 31)],  # Hurricane Harvey (2017)
        [date(2019, 5, 1), date(2019, 4, 30)],  # California Wildfires (2018)
        [date(2017, 9, 1), date(2018, 8, 31)],  # Hurricane Maria (2017)
        [date(2018, 9, 1), date(2019, 8, 31)],  # Hurricane Michael (2018)

        [date(2019, 1, 1), date(2019, 12, 31)],  # Sulawesi Earthquake and Tsunami (2018)
        [date(2015, 4, 1), date(2016, 3, 31)],  # Nepal Earthquake (2015)
        [date(2017, 5, 1), date(2018, 4, 30)],  # Bangladesh Cyclone Mora (2017)
        [date(2018, 7, 1), date(2019, 6, 30)],  # India Kerala Floods (2018)

        [date(2019, 5, 1), date(2020, 4, 30)],  # European Heatwaves (2019)
        [date(2017, 6, 1), date(2018, 5, 31)],  # Portugal Wildfires (2017)
        [date(2014, 5, 1), date(2015, 4, 30)],  # European Floods (2014)
        [date(2018, 7, 1), date(2019, 6, 30)],  # Greek Wildfires (2018)
        [date(2016, 8, 1), date(2017, 7, 31)],  # Italy Earthquakes (2016)

        # Geopolitical Conflicts / Armed Conflicts (US)
        [date(2016, 10, 1), date(2017, 7, 31)],  # Mosul Offensive (2016-2017) - Iraq
        [date(2014, 9, 1), date(2015, 3, 31)],  # Battle of Kobani (2014-2015) - Syria
        [date(2017, 6, 1), date(2017, 10, 31)],  # Battle of Raqqa (June - October 2017) - Syria
        [date(2015, 9, 1), date(2015, 10, 31)],  # Kunduz City Attack (2015) - Afghanistan
        [date(2016, 5, 1), date(2016, 12, 31)],  # Battle of Sirte (2016) - Libya

        # Geopolitical Conflicts / Armed Conflicts (Asia)
        [date(2019, 2, 1), date(2019, 2, 28)],  # "Pulwama attack and Balakot Airstrikes, kashmir conflict (February 2019) - India / Pakistan"
        [date(2016, 11, 1), date(2019, 12, 31)], # "Syrian Civil War, Aleppo Offensive (2016))",
        [date(2018, 6, 1), date(2018, 12, 31)],  # Yemeni Civil War, Battle of Hudaydah (June - December 2018)

        # Geopolitical Conflicts / Armed Conflicts (Europe)
        [date(2014, 2, 1), date(2014, 12, 31)],  # Crimea Annexation and Conflict in Eastern Ukraine (2014)
        [date(2016, 4, 1), date(2016, 4, 30)],  # Nagorno-Karabakh Conflicts (Clashes (2016)) - Armenia-Azerbaijan
]

    if not all_plots:
        # Retrieve terms and title for the given index
        list_of_lists = list_of_lists_environment + list_of_lists_conflict
        titles = titles_environment + titles_conflict

        terms = list_of_lists[index]
        title = titles[index]
        
        # Build the filtering condition based on complex terms logic
        final_condition = build_filter_condition(terms)
        print(f"Event: {title}")

        event_metadata = filtered_df_metadata.filter(final_condition)
        
        if crop_time:
            event_metadata = event_metadata.filter( (pl.col("upload_date") > event_dates[index][0]) & (pl.col("upload_date") < event_dates[index][1]))


        print(f"Related videos found: {event_metadata.shape[0]:,}")

        match grouping_mode:
            case "daily":
                date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
            case "weekly":
                date_counts = (event_metadata
                               .with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date"))
                               .group_by("upload_date")
                               .count()
                               .sort("upload_date"))
            case "monthly":
                date_counts = (event_metadata
                               .with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date"))
                               .group_by("upload_date")
                               .count()
                               .sort("upload_date"))

        # Plot the histogram
        plt.figure(figsize=(16, 6))
        plt.bar(date_counts["upload_date"], (date_counts["count"]), width=3)
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title(f"Upload frequency for videos related to {title}")

        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.xlim(datetime(2013, 1, 1), datetime(2019, 12, 1))
        # plt.ylim(0,10)
        plt.show()
        return event_metadata       ####
    else:
        if event_type == "environment":
            list_of_lists = list_of_lists_environment
            titles = titles_environment
        elif event_type == "conflict":
            list_of_lists = list_of_lists_conflict
            titles = titles_conflict
        else:
            raise KeyError("wrong event type: \"environment\" or \"conflict\" ")

        fig, ax = plt.subplots(len(list_of_lists), 1, figsize=(16, 36), constrained_layout=True)
        for idx in range(len(list_of_lists)):
            terms = list_of_lists[idx]
            title = titles[idx]

            print(f"Event: {title}")

            final_condition = build_filter_condition(terms)
            event_metadata = filtered_df_metadata.filter(final_condition)
            
            if crop_time:
                event_metadata = event_metadata.filter( (pl.col("upload_date") > event_dates[idx][0]) & (pl.col("upload_date") < event_dates[idx][1]))

            print(f"Related videos found: {event_metadata.shape[0]:,}")
            print("------------")

            match grouping_mode:
                case "daily":
                    date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
                case "weekly":
                    date_counts = (event_metadata
                                   .with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date"))
                                   .group_by("upload_date")
                                   .count()
                                   .sort("upload_date"))
                case "monthly":
                    date_counts = (event_metadata
                                   .with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date"))
                                   .group_by("upload_date")
                                   .count()
                                   .sort("upload_date"))

            ax[idx].bar(date_counts["upload_date"], date_counts["count"], width=3)
            ax[idx].set_xlabel("Date")
            ax[idx].set_ylabel("Count")
            ax[idx].set_title(f"Upload frequency for videos related to {title}")

            ax[idx].xaxis.set_major_locator(mdates.YearLocator())
            ax[idx].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax[idx].xaxis.get_majorticklabels(), rotation=90)
            ax[idx].set_xlim(datetime(2013, 1, 1), datetime(2019, 12, 1))
            ax[idx].grid(True)

        plt.show()

def plot_update_freq_v5(index, filtered_df_metadata, all_plots=False, grouping_mode="daily", event_type=None, crop_time=False, generate = False): # added argument crop_time

    event_dates = [
        [date(2017, 8, 1), date(2018, 7, 31)],  # Hurricane Harvey (2017)
        [date(2018, 5, 1), date(2018, 4, 30)],  # California Wildfires (2018)
        [date(2017, 9, 1), date(2018, 8, 31)],  # Hurricane Maria (2017)
        [date(2018, 9, 1), date(2019, 8, 31)],  # Hurricane Michael (2018)

        [date(2019, 1, 1), date(2019, 12, 31)],  # Sulawesi Earthquake and Tsunami (2018)
        [date(2015, 4, 1), date(2016, 3, 31)],  # Nepal Earthquake (2015)
        [date(2017, 5, 1), date(2018, 4, 30)],  # Bangladesh Cyclone Mora (2017)
        [date(2018, 7, 1), date(2019, 6, 30)],  # India Kerala Floods (2018)

        [date(2019, 5, 1), date(2020, 4, 30)],  # European Heatwaves (2019)
        [date(2017, 6, 1), date(2018, 5, 31)],  # Portugal Wildfires (2017)
        [date(2014, 5, 1), date(2015, 4, 30)],  # European Floods (2014)
        [date(2018, 7, 1), date(2019, 6, 30)],  # Greek Wildfires (2018)
        [date(2016, 8, 1), date(2017, 7, 31)],  # Italy Earthquakes (2016)

        # Geopolitical Conflicts / Armed Conflicts (US)
        [date(2016, 10, 1), date(2017, 7, 31)],  # Mosul Offensive (2016-2017) - Iraq
        [date(2014, 9, 1), date(2015, 3, 31)],  # Battle of Kobani (2014-2015) - Syria
        [date(2017, 6, 1), date(2017, 10, 31)],  # Battle of Raqqa (June - October 2017) - Syria
        [date(2015, 9, 1), date(2015, 10, 31)],  # Kunduz City Attack (2015) - Afghanistan
        [date(2016, 5, 1), date(2016, 12, 31)],  # Battle of Sirte (2016) - Libya

        # Geopolitical Conflicts / Armed Conflicts (Asia)
        [date(2019, 2, 1), date(2019, 2, 28)],  # "Pulwama attack and Balakot Airstrikes, kashmir conflict (February 2019) - India / Pakistan"
        [date(2016, 11, 1), date(2019, 12, 31)], # "Syrian Civil War, Aleppo Offensive (2016))",
        [date(2018, 6, 1), date(2018, 12, 31)],  # Yemeni Civil War, Battle of Hudaydah (June - December 2018)

        # Geopolitical Conflicts / Armed Conflicts (Europe)
        [date(2014, 2, 1), date(2014, 12, 31)],  # Crimea Annexation and Conflict in Eastern Ukraine (2014)
        [date(2016, 4, 1), date(2016, 4, 30)],  # Nagorno-Karabakh Conflicts (Clashes (2016)) - Armenia-Azerbaijan
]

    if not all_plots:
        # Retrieve terms and title for the given index
        list_of_lists = list_of_lists_environment + list_of_lists_conflict
        titles = titles_environment + titles_conflict

        terms = list_of_lists[index]
        title = titles[index]
        
        # Build the filtering condition based on complex terms logic
        final_condition = build_filter_condition(terms)
        print(f"Event: {title}")

        event_metadata = filtered_df_metadata.filter(final_condition)
        
        if crop_time:
            event_metadata = event_metadata.filter( (pl.col("upload_date") > event_dates[index][0]) & (pl.col("upload_date") < event_dates[index][1]))

        print(f"Related videos found: {event_metadata.shape[0]:,}")

        match grouping_mode:
            case "daily":
                date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
            case "weekly":
                date_counts = (event_metadata
                               .with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date"))
                               .group_by("upload_date")
                               .count()
                               .sort("upload_date"))
            case "monthly":
                date_counts = (event_metadata
                               .with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date"))
                               .group_by("upload_date")
                               .count()
                               .sort("upload_date"))

        # Plot the histogram
        plt.figure(figsize=(16, 6))
        plt.bar(date_counts["upload_date"], (date_counts["count"]), width=3)
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title(f"Upload frequency for videos related to {title}")

        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.xlim(datetime(2013, 1, 1), datetime(2019, 12, 1))
        # plt.ylim(0,10)
        plt.show()
        return event_metadata       ####
    else:

        # if event_type == "environment":
        #     list_of_lists = list_of_lists_environment
        #     titles = titles_environment
        # elif event_type == "conflict":
        #     list_of_lists = list_of_lists_conflict
        #     titles = titles_environment
        # else:
        #     raise KeyError("wrong event type: \"environment\" or \"conflict\" ")
        
        list_of_lists = list_of_lists_environment + list_of_lists_conflict
        titles = titles_environment + titles_environment
        
        fig, ax = plt.subplots(len(list_of_lists), 1, figsize=(16, 36), constrained_layout=True)

        # schema = {col: dtype for col, dtype in zip(filtered_df_metadata.columns, filtered_df_metadata.dtypes)}
        # schema["event"] = pl.Utf8 
        # total_output_df = pl.DataFrame(schema=schema)
        # for idx in range(len(list_of_lists)):
        #     terms = list_of_lists[idx]
        #     title = titles[idx]

        #     print(f"Event: {title}")

        #     final_condition = build_filter_condition(terms)
        #     event_metadata = filtered_df_metadata.filter(final_condition)
            
        #     if crop_time:
        #         event_metadata = event_metadata.filter( (pl.col("upload_date") > event_dates[idx][0]) & (pl.col("upload_date") < event_dates[idx][1]))

        #     print(f"Related videos found: {event_metadata.shape[0]:,}")
        #     print("------------")

        #     output = event_metadata.clone()
        #     if generate:
        #         output = output.with_columns(pl.lit(title).alias("event"))
        #         total_output_df = pl.concat([total_output_df, output], how = "vertical")

        # Initialize total_output_df with additional columns
        schema = {col: dtype for col, dtype in zip(filtered_df_metadata.columns, filtered_df_metadata.dtypes)}
        schema["event"] = pl.Utf8
        schema["region"] = pl.Utf8
        schema["event_type"] = pl.Utf8

        total_output_df = pl.DataFrame(schema=schema)

        # Main processing loop
        for idx in range(len(list_of_lists)):
            terms = list_of_lists[idx]
            title = titles[idx]

            print(f"Event: {title}")

            final_condition = build_filter_condition(terms)
            event_metadata = filtered_df_metadata.filter(final_condition)

            if crop_time:
                event_metadata = event_metadata.filter(
                    (pl.col("upload_date") > event_dates[idx][0]) & 
                    (pl.col("upload_date") < event_dates[idx][1])
                )

            print(f"Related videos found: {event_metadata.shape[0]:,}")
            print("------------")

            # Determine event metadata based on idx
            if idx <= 3:  # US Environment Events
                region = "US"
                event_type = "environmental"
            elif 4 <= idx <= 7:  # Asia Environment Events
                region = "Asia"
                event_type = "environmental"
            elif 8 <= idx <= 12:  # Europe Environment Events
                region = "Europe"
                event_type = "environmental"
            elif 13 <= idx <= 17:  # US Conflicts
                region = "US"
                event_type = "geopolitical conflict"
            elif 18 <= idx <= 20:  # Asia Conflicts
                region = "Asia"
                event_type = "geopolitical conflict"
            elif 21 <= idx <= 22:  # Europe Conflicts
                region = "Europe"
                event_type = "geopolitical conflict"
            else:
                region = "Unknown"
                event_type = "Unknown"

            # Clone and extend the DataFrame
            output = event_metadata.clone()
            if generate:
                output = output.with_columns([
                    pl.lit(title).alias("event"),
                    pl.lit(region).alias("region"),
                    pl.lit(event_type).alias("event_type")
                ])
                total_output_df = pl.concat([total_output_df, output], how="vertical")

            match grouping_mode:
                case "daily":
                    date_counts = event_metadata.group_by("upload_date").count().sort("upload_date")
                case "weekly":
                    date_counts = (event_metadata
                                   .with_columns(pl.col("upload_date").dt.truncate("1w").alias("upload_date"))
                                   .group_by("upload_date")
                                   .count()
                                   .sort("upload_date"))
                case "monthly":
                    date_counts = (event_metadata
                                   .with_columns(pl.col("upload_date").dt.truncate("1mo").alias("upload_date"))
                                   .group_by("upload_date")
                                   .count()
                                   .sort("upload_date"))

            ax[idx].bar(date_counts["upload_date"], date_counts["count"], width=3)
            ax[idx].set_xlabel("Date")
            ax[idx].set_ylabel("Count")
            ax[idx].set_title(f"Upload frequency for videos related to {title}")

            ax[idx].xaxis.set_major_locator(mdates.YearLocator())
            ax[idx].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax[idx].xaxis.get_majorticklabels(), rotation=90)
            ax[idx].set_xlim(datetime(2013, 1, 1), datetime(2019, 12, 1))
            ax[idx].grid(True)

        plt.show()
        if generate:
            return total_output_df