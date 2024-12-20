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


hurricane_harvey_keywords = [         
    ["Hurricane", "Harvey"], 
    ["Texas", "flood"], 
    ["Houston", "flood"], 
    ["Harvey", "storm"], 
    ["Harvey", "disaster"], 
    ["Houston", "hurricane"], 
    ["Harvey", "relief", "efforts"],
    ["harvey", "houston"],
]

california_wildfires_keywords = [    
    ["California", "wildfire"], 
    ["California", "firestorm"], 
    ["California", "fire"],

    ["Camp", "Fire"], 
    ["Paradise", "fire"],     
    ["Camp", "wildfire"], 
    ["Paradise", "wildfire"],  
]

hurricane_maria_keywords = [         
    ["Hurricane", "Maria"], 
    ["Puerto", "Rico", "maria"], 
    ["Maria", "disaster"], 
    ["Maria", "flood"], 
    ["Maria", "aftermath"]
]

hurricane_michael_keywords = [        
    ["michael", "hurricane"], 
]

sulawesi_earthquake_tsunami_keywords = [   
    ["Sulawesi", "quake"], 
    ["Sulawesi", "tsunami"], 
    ["Indonesia", "quake"], 
    ["Indonesia", "tsunami"], 
]

nepal_earthquake_keywords = [               
    ["Kathmandu", "quake"], 
    ["Nepal", "quake"]
]

bangladesh_cyclone_mora_keywords = [        
    ["Cyclon", "Mora"],
]

india_floods_keywords = [     
    ["Kerala", "flood"]
]

heatwaves_2019_keywords = [    
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

portugal_wildfires_keywords = [        
    ["Portugal", "fire"], 
    ["Pedrógão", "fire"], 
]

european_floods_2014_keywords = [    
    ["Europe", "flood"], 
    ["Balkan", "flood"], 
    ["Bosnia", "flood"], 
    ["Serbia", "flood"], 
    ["Croatia", "flood"], 
    ["Romania", "flood"], 
    ["Bulgaria", "flood"]
]

greek_wildfires_keywords = [        
    ["Athens", "fire"], 
    ["Greece", "fire"],
    ["Greek", "fire"],
    ["attica", "wildfire"]
]

italy_earthquakes_keywords = [       
    ["Amatrice", "quake"], 
    ["italy", "quake"],
    ["italian", "quake"],
]

# Keyword lists for geopolitical and armed conflicts

# US Involvement Conflicts
mosul_offensive_keywords = [        
    ["Mosul", "Offensive"], 
    ["Mosul", "battle"], 
    ["Mosul", "US"], 
    ["Mosul", "strike"], 
    ["Mosul", "fight"], 
    ["Mosul", "combat"], 
    ["Mosul", "capture"],
    ["Mosul", "attack"]
]

battle_of_kobani_keywords = [         
    ["kobani"]
]

battle_of_raqqa_keywords = [         
    ["raqqa"]
]

kunduz_city_attack_keywords = [        
    ['kunduz']
]

battle_of_sirte_keywords = [         
    ["Sirte"]
]

# Asia Conflicts
india_pakistan_conflict_keywords = [    
    ["Pulwama"],
    ["Balakot"],
]

syrian_civil_war_keywords = [            
    ["Aleppo"]
]

yemeni_civil_war_keywords = [             
    ['yemen'],
    ['hudaida'],
    ['hudaydah']
]

nagorno_karabakh_conflicts_keywords = [     
    ['nagorno'],
    ["karabakh"],
    ["armenia", "azerbaijan"]
]

# Europe Conflicts
crimea_annexation_keywords = [            
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
    "India Floods in Kerela (2018)", 

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
    "Battle of Raqqa (Jun - Oct 2017) - Syria",
    "Kunduz City Attack (2015) - Afghanistan",
    "Battle of Sirte (2016) - Libya",
    
    # Geopolitical Conflicts / Armed Conflicts (Asia)
    "Pulwama attack & Balakot Airstrikes (Feb 2019) - India/Pakistan",
    "Syrian Civil War, Aleppo Offensive (2016)",
    "Yemeni Civil War, battle of Hudayah (Jun - Dec 2018)",
    
    # Geopolitical Conflicts / Armed Conflicts (Europe)
    "Crimea Annexation & Conflict in Eastern Ukraine (2014)",
    "Nagorno-Karabakh Clashes (2016) - Armenia-Azerbaijan",
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


def plot_update_freq(index, filtered_df_metadata, all_plots=False, grouping_mode="daily", event_type=None, crop_time=False, generate = False): # added argument crop_time

    event_dates = [
        [date(2017, 8, 1), date(2018, 7, 30)],  # Hurricane Harvey (Aug-Sep 2017)
        [date(2018, 6, 1), date(2019, 2, 1)],  # California Wildfires (Jul-Nov 2018)
        [date(2017, 8, 1), date(2018, 9, 1)],  # Hurricane Maria (Sep 2017 - Feb 2018)
        [date(2018, 9, 1), date(2019, 3, 1)],  # Hurricane Michael (Oct 2018 - Jan 2019)
        [date(2018, 9, 26), date(2018, 10, 20)],  # Sulawesi Earthquake and Tsunami (Sep-Dec 2018)

        [date(2015, 4, 23), date(2015, 7, 1)],  # Nepal Earthquake (Apr-May 2015)
        [date(2017, 5, 28), date(2017, 7, 1)],  # Bangladesh Cyclone Mora (May-Jun 2017)
        [date(2018, 8, 1), date(2018, 8, 31)],  # India Kerala Floods (Aug 2018)

        [date(2019, 6, 1), date(2019, 8, 31)],  # European Heatwaves (Jun-Aug 2019)
        [date(2017, 6, 17), date(2017, 10, 31)],  # Portugal Wildfires (Jun-Oct 2017)
        [date(2014, 5, 10), date(2014, 6, 15)],  # European Floods (May-Jun 2014)

        [date(2018, 7, 20), date(2018, 8, 15)],  # Greek Wildfires (July 2018)
        [date(2016, 8, 21), date(2016, 9, 15)],  # Italy Earthquakes (2016)

        # Geopolitical Conflicts / Armed Conflicts (US)
        [date(2016, 10, 13), date(2017, 8, 5)],  # Mosul Offensive (Oct 2016 - Jul 2017) - Iraq
        [date(2014, 9, 13), date(2015, 3, 10)],  # Battle of Kobani (Sep 2014 - Mar 2015) - Syria
        [date(2017, 6, 4), date(2017, 10, 30)],  # Battle of Raqqa (Jun-Oct 2017) - Syria
        [date(2015, 9, 25), date(2015, 11, 1)],  # Kunduz City Attack (Sep-Oct 2015) - Afghanistan
        [date(2016, 5, 9), date(2016, 12, 30)],  # Battle of Sirte (May-Dec 2016) - Libya

        # Geopolitical Conflicts / Armed Conflicts (Asia)
        [date(2019, 2, 14), date(2019, 5, 10)],  # Pulwama attack and Balakot Airstrikes (Feb 2019) - India/Pakistan
        [date(2016, 6, 1), date(2016, 12, 31)],  # Syrian Civil War, Aleppo Offensive (Nov 2016 - Dec 2019)
        [date(2018, 6, 10), date(2018, 12, 25)],  # Yemeni Civil War, Battle of Hudaydah (Jun-Dec 2018)

        # Geopolitical Conflicts / Armed Conflicts (Europe)
        [date(2014, 2, 10), date(2014, 12, 31)],  # Crimea Annexation and Conflict in Eastern Ukraine (Feb-Dec 2014)
        [date(2016, 3, 28), date(2016, 4, 8)],  # Nagorno-Karabakh Conflicts (Clashes Apr 2016) - Armenia-Azerbaijan
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
        return event_metadata
    else:

        
        list_of_lists = list_of_lists_environment + list_of_lists_conflict
        titles = titles_environment + titles_conflict
        
        fig, ax = plt.subplots(len(list_of_lists), 1, figsize=(16, 36), constrained_layout=True)

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
                event_type = "geopolitical"
            elif 18 <= idx <= 20:  # Asia Conflicts
                region = "Asia"
                event_type = "geopolitical"
            elif 21 <= idx <= 22:  # Europe Conflicts
                region = "Europe"
                event_type = "geopolitical"
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
            ax[idx].set_title(f"Upload frequency for videos related to {title}, videos found: {event_metadata.shape[0]:,}")

            ax[idx].xaxis.set_major_locator(mdates.YearLocator())
            ax[idx].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax[idx].xaxis.get_majorticklabels(), rotation=90)
            ax[idx].set_xlim(datetime(2013, 1, 1), datetime(2019, 12, 1))
            ax[idx].grid(True)

        plt.show()
        if generate:
            return total_output_df