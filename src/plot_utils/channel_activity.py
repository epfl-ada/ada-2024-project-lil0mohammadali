import polars as pl
from ..scripts import filters
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..utils.keywords import get_event_metadata
from ..utils.analysis_tools import create_video_features_dataframe
import numpy as np


def plot_channel_activity(df_channels: pl.DataFrame, df_timeseries: pl.DataFrame, 
                          save_path: str = None, show: bool = True):
    """
    Plot the average activity of channels in the category 'News & Politics' and display
      a list of channels with an average activity > 56.

      Args:
        df_channels (pl.DataFrame): df_channels_en unfiltered
        df_timeseries (pl.DataFrame): df_timeseries unfiltered
        save_path (str): path to save the plot as an HTML file
        show (bool): whether to display the plot
    """

    filtered_df_ch = filters.filter_df(df_channels, column_name="category_cc", 
                                    value="News & Politics", cmpstr="==")

    filtered_df_timeseries = filters.filter_df_isin(df_timeseries, column_name="channel", 
                                                    values=filtered_df_ch["channel"])

    grouped_df = filtered_df_timeseries.group_by('channel').agg(pl.col('activity').mean().alias('mean_activity'))


    merged_df = filtered_df_ch.join(grouped_df, on='channel', how='inner')

    # divide activity by 14 to get the average activity per day
    merged_df = merged_df.with_columns((pl.col('mean_activity') / 14))

    filtered_df_ch = filters.filter_df(merged_df, "mean_activity", 4, ">")
    filtered_df_ch = filtered_df_ch.sort(by="mean_activity", descending=True)

    # TODO: list with links to channels --> not working yet
    youtube_urls = [
        f'<a href="https://www.youtube.com/channel/{channel_id}>{channel_name}</a>'
        for channel_id, channel_name in zip(filtered_df_ch['channel'], filtered_df_ch['name_cc'])
    ]

    nb_channels = len(filtered_df_ch)

    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Histogram of average activity (videos/day)", 
                                        f"{nb_channels} channels above cutoff"),
                        specs=[[{"type": "histogram"}, {"type": "table"}]],
                        column_widths=[0.5, 0.5])
    fig.add_trace(go.Histogram(x=merged_df['mean_activity'], nbinsx=100, name="Activity Distribution"), row=1, col=1)

    fig.update_layout(
    xaxis_title="Average nb of videos per day",  # X-axis label
    yaxis_title="Count",  # Y-axis label
)

    # display cutoff line
    fig.update_layout(
        shapes=[
            dict(
                type='line', 
                x0=4, x1=4, 
                y0=0, y1=1, 
                yref='paper',  # The line will extend across the entire plot area
                line=dict(color='red', dash='dot', width=2),
                name="Cutoff"  # Name for the legend entry
            )
        ],
        legend=dict(
            itemsizing='constant',
            tracegroupgap=0,
            x=0.35, 
            y=0.95,  
            bgcolor="rgba(255, 255, 255, 0)",  
            bordercolor="black",
            borderwidth=1
        )
    )

    # fig.add_trace(go.Scatter(
    #     x=[None], y=[None],  
    #     mode='lines',
    #     name='Cutoff',  
    #     line=dict(color='red', dash='dot')
    # ))
    # log scale for y-axis
    fig.update_layout(yaxis_type="log")

    fig.add_trace(go.Table(header=dict(values=['Channel', 'Mean activity']),
                    cells=dict(values=[filtered_df_ch['name_cc'], 
                                       filtered_df_ch['mean_activity'].round(0)], 
                                align='left', 
                                font=dict(size=12),  
                                height=30  
                            )), row=1, col=2)



    if show:
        fig.show()
    if save_path:
        fig.write_html(save_path)


def plot_video_metrics(metadata: pl.DataFrame, timeseries_df: pl.DataFrame, save_path: str = None, show: bool = True):
    """
    Plot the video metrics based on event types and origin

      Args:
        df_channels (pl.DataFrame): df_channels_en unfiltered
        df_timeseries (pl.DataFrame): df_timeseries unfiltered
        save_path (str): path to save the plot as an HTML file
        show (bool): whether to display the plot
    """

    event_type = ['geopolitical', 'environmental']
    geographical_location = ['us', 'eu', 'asia']
    list_of_events = ['Event 0', 'Event 1', 'Event 2', 'Event 3', 'Event 4', 'Event 5', 'Event 6', 'Event 7', 'Event 8', 'Event 9', 'Event 10', 'Event 11']
    events = [1,2]
    # plot settings
    vid_feature_columns = ['duration', 'mean_delta_videos', 'std_delta_videos', 'capitalisation_ratio', 'is_live']
    subplot_titles_ = ['Duration (sec)', 'Mean Delta Videos', 'Std Delta Videos', 'Capitalization Ratio', 
                    'Proportion of live<br> videos in US', 'Proportion of live<br> videos in eu', 'Proportion of live<br> videos in asia']
    num_subplots = 4*3+3
    row_num = 2
    col_num = 4 

    buttons = []
    # separated by event type
    fig = make_subplots(rows=row_num, cols=col_num, 
                        subplot_titles=subplot_titles_, 
                        horizontal_spacing = 0.1, vertical_spacing=0.15,
                        specs=[[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}, {'type': 'box'}],  # First row for box plots
                                [{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}, None]],
                        column_widths=[0.25, 0.25, 0.25, 0.25,],  # Equal column widths
                        row_heights=[0.5, 0.5] ) # Equal row heights ) # All subplots are 'xy' type by default)

    annotations = []
    for i, title in enumerate(subplot_titles_):
        row = (i // col_num) + 1
        col = (i % col_num) + 1
        annotations.append(dict(
            # x= col / col_num - 1 / (2 * col_num),
            y = 1 - (row - 1) / row_num + 0.02 if row == 1 else 1 - (row - 1) / row_num - 0.08,
            text=title,
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            font=dict(size=16)
        ))


    # Customize layout
    fig.update_layout(
        yaxis_title="Values",
        # boxmode='group',  # Grouped box plots
        height=600,
        width=800,
        showlegend=True,
        annotations = annotations,
        xaxis=dict(showticklabels=True),  # For the first subplot
        xaxis2=dict(showticklabels=True),  # For the second subplot
        xaxis3=dict(showticklabels=True),  # For the third subplot
        xaxis4=dict(showticklabels=True),  # For the fourth subplot
        xaxis5=dict(showticklabels=True),  # For the fifth subplot
        yaxis1=dict(
            tickvals=[1e-1, 1, 10, 100, 1000, 10000, 100000],  # Start closer to 1
            ticktext=["0", "1", "10", "100", "1k", "10k", "100k"],  # Label 1e-2 as "0"
            type='log',  # Logarithmic scale
            autorange=False,
            range=[-1, 5]  # Adjust the range accordingly
        ),
        # ),
        yaxis4=dict(
            tickvals=[1e-2, 1e-1, 1, 10, 100],  # Start closer to 1
            ticktext=["0", "0.1", "1", "10", '100'],  # Label 1e-2 as "0"
            type='log',  # Logarithmic scale
            autorange=False,
            range=[-2, 2]  # Adjust the range accordingly
        )
    )

    for i, event in enumerate(events):
        event_metadata = get_event_metadata(metadata, event)
        # create the video features dataframe for the event from the metadata and timeseries dataframes

        ## MAY REMOVE THIS BECUSE TIME SERIES DATA MAY NOT BE PLOTTED
        vid_features = create_video_features_dataframe(event_metadata, timeseries_df)

        ############### FOR TESTING, REMOVE WITH REAL DATAFRAME #################
        regions = np.random.choice(['us', 'eu', 'asia'], size=vid_features.height)
        vid_features = vid_features.with_columns(pl.Series('region', regions))
        #########################################################################

        fig.update_layout(
            title=f"Box Plot of Video Metrics for {event_type[i]} Events",
        )

        visible = (i == 0)
        # Loop through columns and add a box plot for each
        for idx, column in enumerate(vid_feature_columns):
            row = (idx // col_num) + 1  # Calculate row number
            col = (idx % col_num) + 1   # Calculate column number
            # Add trace to the subplot
            # plotting the circles
            if row == 2 and col == 1:
                live_values_df=vid_features.filter(pl.col('region') == 'us')['is_live'].value_counts()
                live_values_df = live_values_df.to_pandas()
                live_values_df['is_live'] = live_values_df['is_live'].map({0: 'Not Live', 1: 'Live'})
                fig.add_trace(
                    go.Pie(labels=live_values_df['is_live'], 
                        values=live_values_df['count'], 
                        showlegend=True),
                    row=row,
                    col=col,
                )
                # print("circle plot for eu")
                live_values_df=vid_features.filter(pl.col('region') == 'eu')['is_live'].value_counts()
                live_values_df = live_values_df.to_pandas()
                live_values_df['is_live'] = live_values_df['is_live'].map({0: 'Not Live', 1: 'Live'})
                fig.add_trace(
                    go.Pie(labels=live_values_df['is_live'], 
                        values=live_values_df['count'], 
                        showlegend=True),
                    row=row,
                    col=col+1,
                )
                # print("circle plot for asia")
                live_values_df=vid_features.filter(pl.col('region') == 'asia')['is_live'].value_counts()
                live_values_df = live_values_df.to_pandas()
                live_values_df['is_live'] = live_values_df['is_live'].map({0: 'Not Live', 1: 'Live'})
                fig.add_trace(
                    go.Pie(labels=live_values_df['is_live'], 
                        values=live_values_df['count'], 
                        showlegend=True),
                    row=row,
                    col=col+2,
                )
            else:
                # plotting box plots comparing geographical location
                for region in geographical_location:
                    # print(vid_features)
                    # print(region)
                    region_features = vid_features.filter(pl.col('region') == region)[column].to_list()
                    # print(region_features)
                    fig.add_trace(
                        go.Box(
                            y=region_features,
                            name=f"{region}",
                            # jitter=0.3,
                            # pointpos=-2.0,
                            # width=0.001,
                            showlegend=False,
                            boxpoints=False # suspectedoutliers, all, outliers
                            # boxmean='sd'  # Show mean and standard deviation
                        ),
                    row=row,
                    col=col,
                ) 
    
        buttons.append(dict(
            args=[{'visible': [False]*i*num_subplots + [True]*num_subplots + [False]*(len(list_of_events)-i-1)*num_subplots}],
            label=list_of_events[i],
            method="update"
        ))

    initial_visibility = [True]*num_subplots + [False]*(len(events)-1)*num_subplots
    for i in range(len(fig.data)):
        fig.data[i].visible = initial_visibility[i]    

    fig.update_layout(
        title=f"Box Plot of Video Metrics for {event_type[0]} Events",
        title_x = 0.5,
        title_xanchor='center',
        updatemenus=[
            dict(
                buttons=list(buttons),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="right",
                y=1.2,
                yanchor="top"
            )
        ],

        # placing legend for pie plot
        legend=dict(
            x=0.8,  # origin is bottom left
            y=0.3,   
            xanchor='left',  # anchor the legend's x-position to the left side
            yanchor='middle',  # anchor the legend's y-position to the middle
            orientation='v'  # legend items stacked vertically
        ),
        hovermode='x'
    )

    if show:
        fig.show(scrollZoom=False)
    if save_path:
        fig.write_html(save_path)


if __name__ == "__main__":
    path_df_channels_en = 'data/df_channels_en.tsv'

    path_df_timeseries = 'data/df_timeseries_en.tsv'

    path_yt_metadata_feather = 'data/yt_metadata_helper.feather'
    path_yt_metadata_feather_filtered = 'data/filtered_yt_metadata_helper.feather.csv'

    path_yt_metadata = 'data/yt_metadata_en.jsonl'
    path_yt_metadata_filtered = 'data/filtered_yt_metadata.csv'

    path_final_channels = 'data/final_channels.csv'
    path_final_timeseries = 'data/final_timeseries.csv'
    path_final_yt_metadata_feather = 'data/final_yt_metadata_helper.csv'
    path_final_yt_metadata = 'data/final_yt_metadata.csv'

    df_channels = pl.read_csv(path_final_channels)
    df_timeseries = pl.read_csv(path_final_timeseries)

    plot_channel_activity(df_channels, df_timeseries, show=True)