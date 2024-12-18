# this file contains the updated filtering function with only natural disasters and geopolitical conflicts

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


hurricane_harvey_keywords = [           # ok
    ["Hurricane", "Harvey"], 
    ["Texas", "flood"], 
    ["Houston", "flood"], 
    ["Harvey", "storm"], 
    ["Harvey", "disaster"], 
    ["Houston", "hurricane"], 
    ["Harvey", "relief", "efforts"],
    ["harvey", "houston"]
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

southeast_asian_haze_keywords = [               ### difficult to locate or No reporting
    # ["Southeast", "Asia", "haze"], 
    # ["southeast", "asia"]  
    ["asia", "haze"]
    # ["transboundary", "haze"], 
    # ["Indonesia", "fires"], 
    # ["agricultural", "fires"], 
    # ["hazardous", "air", "quality"], 
    # ["air", "pollution"], 
    # ["haze", "emergency"], 
    # ["Southeast", "Asia", "smog"]
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

hudaydah_offensive_keywords = [     
    # ["Hudaydah", "Offensive"], 
    # ["Yemen", "Hudaydah", "battle"], 
    ["Saudi-led", "coalition", "Yemen"], 
    # ["Hudaydah", "port", "battle"], 
    ["US", "support", "Yemen"], 
    # ["Houthi", "rebels", "Hudaydah"]
    ["Hudaydah"]
]

kunduz_city_attack_keywords = [
    ["Kunduz", "attack"], 
    ["Afghanistan", "Kunduz"], 
    ["Taliban", "Kunduz", "battle"], 
    ["US", "airstrikes", "Kunduz"], 
    ["Kunduz", "counter-offensive"], 
    ["Kunduz", "capture"]
]

battle_of_sirte_keywords = [
    ["Battle", "Sirte"], 
    ["Libya", "Sirte"], 
    ["Sirte", "ISIS"], 
    ["US", "airstrikes", "Sirte"], 
    ["street", "fighting", "Sirte"], 
    ["Sirte", "recapture"]
]

# Asia Conflicts
india_pakistan_conflict_keywords = [        # ok
    ["Pulwama"],
    ["Balakot"],
    # ["kashmir"]
]

syrian_civil_war_keywords = [               # ok
    ["Aleppo"]
]

yemeni_civil_war_keywords = [               # search for particular events / clashes
    # ["Yemeni", "Civil", "War"], 
    # ["Yemen", "conflict"], 
    # ["Houthi", "rebels"], 
    # ["Saudi-led", "coalition"], 
    # ["Yemen", "humanitarian", "crisis"], 
    # ["Yemen", "civil", "war"]
    ['yemen']
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

kosovo_serbia_tensions_keywords = [        # not actually clashes??
    # ["Kosovo", "Serbia", "tensions"], 
    # ["border", "clashes"], 
    # ["Kosovo", "Serbia", "conflict"], 
    # ["political", "dispute", "Kosovo"], 
    # ["balkan", "armed", "incidents"], 
    # ["balkan", "territorial"],
    # # ["kosovo", "serbia"],
    # ["kosovo", "serb"],
    # # ["serbia", "conflict"],
    # ["kosovo", "crisis"],
    # ["kosovo", "raid"],
    # ["kosovo", "tesnsion"],
    # ["kosovo", "conflict"],
    # ["balkan", "dispute"],
    # ["north", "kosovo"],
    # ["kosovo", "police"],
    # ["pristina"],
    # ["kosovo", "border"]
    ["hungar"]
]

# ukraine_conflict_keywords = [               # overlaps with the annexation
#     ["Ukraine", "conflict"], 
#     ["War", "Donbas"], 
#     ["Ukraine", "Donbas"], 
#     ["Russian-backed", "separatists"], 
#     ["Eastern", "Ukraine", "war"], 
#     ["Ukraine", "international", "sanctions"]
# ]

kumanovo_keywords = [               # WTF
    ["kumanovo"],

    ["macedonia", "clash"],
    # ["macedonia", "police"]
    ["NLA", "SDSM"],
    # ["national liberation army"],

    # ["SDSM"],
    # ["social democratic union of macedonia"]
]

Isani_flat_siege_keywords = [       # WTF 2, no videos or unrelated
    # ["Tbilisi"],
    ["Isani"]
]

turkey_northern_syria_keywords = [          # europe or middle east?
    ["Turkey", "military", "operations"], 
    ["Northern", "Syria", "battle"], 
    ["Operation", "Olive", "Branch"], 
    ["Afrin", "offensive"], 
    ["Turkey", "Kurdish", "forces"], 
    ["YPG", "militia", "conflict"],
    ["turkey", "syria"]
]

list_of_lists_environment = [
    # Natural Disasters
    hurricane_harvey_keywords, 
    california_wildfires_keywords, 
    hurricane_maria_keywords, 
    hurricane_michael_keywords,
    sulawesi_earthquake_tsunami_keywords, 
    southeast_asian_haze_keywords, 
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
    hudaydah_offensive_keywords, 
    kunduz_city_attack_keywords, 
    battle_of_sirte_keywords,

    # Geopolitical and Armed Conflicts (Asia)
    india_pakistan_conflict_keywords, 
    syrian_civil_war_keywords, 
    yemeni_civil_war_keywords, 
    nagorno_karabakh_conflicts_keywords, 

    # Geopolitical and Armed Conflicts (Europe)
    crimea_annexation_keywords, 
    kosovo_serbia_tensions_keywords, 
    kumanovo_keywords, 
    Isani_flat_siege_keywords,
    turkey_northern_syria_keywords
]

titles_environment = [
    # US
    "Hurricane Harvey (2017)", 
    "California Wildfires (2018)", 
    "Hurricane Maria (2017)", 
    "Hurricane Michael (2018)", 

    # Asia
    "Sulawesi Earthquake and Tsunami (2018)", 
    "Southeast Asian Haze (2015)", 
    "Nepal Earthquake (2015)", 
    "Bangladesh Cyclone Mora (2017)", 
    "India Floods (2018)", 

    # Europe
    "Heatwaves (2019)", 
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
    "Hudaydah Offensive (2018) - Yemen",
    "Kunduz City Attack (2015) - Afghanistan",
    "Battle of Sirte (2016) - Libya",
    
    # Geopolitical Conflicts / Armed Conflicts (Asia)
    "India-Pakistan Conflict / Kashmir Conflict (Pulwama and Balakot Airstrikes) (February 2019)",
    "Syrian Civil War (Aleppo Offensive (2016), Idlib Campaign (2019))",
    "Yemeni Civil War (2015-2019) - West Asia",
    "Nagorno-Karabakh Conflicts (Clashes (2016)) - Armenia-Azerbaijan",
    
    # Geopolitical Conflicts / Armed Conflicts (Europe)
    "Crimea Annexation and Conflict in Eastern Ukraine (2014)",
    "Kosovo–Serbia Tensions and Border Clashes (2018-2019)",
    # "Ukraine Conflict - War in Donbas (2014-2019)",
    "Kumanovo clashes (2015) - Macedonia",
    "Isani flat siege (2017) - Georgia",
    "Turkey’s Military Operations in Northern Syria (Operation Olive Branch, 2018)"
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

    event_date_ranges = {
        "Hurricane Harvey (2017)": ("2017-08-01", "2018-07-31"),
        "California Wildfires (2018)": ("2018-05-01", "2019-04-30"),
        "Hurricane Maria (2017)": ("2017-09-01", "2018-08-31"),
        "Hurricane Michael (2018)": ("2018-09-01", "2019-08-31"),

        "Sulawesi Earthquake and Tsunami (2018)": ("2018-09-01", "2019-08-31"),
        "Southeast Asian Haze (2015)": ("2015-06-01", "2016-05-31"),
        "Nepal Earthquake (2015)": ("2015-04-01", "2016-03-31"),
        "Bangladesh Cyclone Mora (2017)": ("2017-05-01", "2018-04-30"),
        "India Floods (2018)": ("2018-07-01", "2019-06-30"),

        "Heatwaves (2019)": ("2019-05-01", "2020-04-30"),
        "Portugal Wildfires (2017)": ("2017-06-01", "2018-05-31"),
        "European Floods (2014)": ("2014-05-01", "2015-04-30"),
        "Greek Wildfires (2018)": ("2018-07-01", "2019-06-30"),
        "Italy Earthquakes (2016)": ("2016-08-01", "2017-07-31")
    }

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
        if crop_time:
            
            event_metadata = event_metadata.filter(pl.col("upload_date") > start_date & pl.col("upload_date") > start_date)

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
            
            # if start_date:
                # event_metadata = event_metadata.filter(pl.col("upload_date") > start_date)

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