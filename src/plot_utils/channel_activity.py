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

def add_pie_chart_for_metric(fig: go.Figure, 
                             video_metadata: pl.DataFrame, 
                             metric_to_plot: str, 
                             metric_name: str,
                             metrics_to_group_by: None, 
                             element_to_group_by: str,
                             row: int,
                             column: int):
    '''
        Pie chart of percent variables
        Args:
            fig: is figure to plot on
            video_metadata: is dataframe
            metric_to_plot: is the data we want to plot (breaking, footage, update)
            metric_name: (name to use for legend)
            metrics_to_group_by: list of elements to group by (list of regions or list of events)
            element_to_group_by: is the grouping (by event or region)
            row: is row to plot on
            col: is col to plot on
    '''

    for i,metric in enumerate(metrics_to_group_by):
        live_values_df=video_metadata.filter(pl.col(element_to_group_by) == metric)[metric_to_plot].value_counts()
        live_values_df = live_values_df.to_pandas()
        live_values_df[metric_to_plot] = live_values_df[metric_to_plot].map({False: f'Not {metric_name}', True: f'{metric_name}'})
        
        fig.add_trace(
            go.Pie(labels=live_values_df[metric_to_plot], 
                values=live_values_df['count'], 
                showlegend=True),
            row=row,
            col=column+i,
        )
    return fig

def plot_video_metrics_event_type(video_metadata: pl.DataFrame, save_path: str = None, show: bool = True):
    """
    Plot the video metrics distinguished by event types, and plots the distribution based on region

      Args:
        df_channels (pl.DataFrame): df_channels_en unfiltered
        df_timeseries (pl.DataFrame): df_timeseries unfiltered
        save_path (str): path to save the plot as an HTML file
        show (bool): whether to display the plot
    """

    event_type = ['geopolitical', 'environmental']
    geographical_location = ['US', 'Europe', 'Asia']
    # plot settings
    vid_feature_columns = ['duration', 'channel_activity', 'subjectivity', 'capitalisation_ratio', 'is_footage']
    subplot_titles_ = ['Duration (sec)', 'Channel Activity', 'Subjectivity', 'Capitalization Ratio', 
                        'Footage %: US', 'Footage %: Europe ', 'Footage %: Asia', 
                        'Update %: US', 'Update %: Europe ', 'Update %: Asia', ' ',
                        'Breaking %: US', 'Breaking %: Europe ', 'Breaking %: Asia']
    
    # subplot_titles_ = ['Duration (sec)', 'Channel Activity', 'Subjectivity', 
    #              'Capitalization Ratio', ' ', 
    #              'Is Footage %:<br>Geopolitical', 'Is Footage %:<br>Environmental']
    num_subplots = 4*3+3*3
    row_num = 4
    col_num = 4 

    buttons = []
    # separated by event type
    fig = make_subplots(rows=row_num, cols=col_num, 
                        subplot_titles=subplot_titles_, 
                        horizontal_spacing = 0.08, vertical_spacing=0.1,
                        specs=[[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}, {'type': 'box'}],  # First row for box plots
                                [{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}, None],
                                [{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}, None],
                                [{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}, None]],
                        column_widths=[0.25, 0.25, 0.25, 0.25,],  # Equal column widths
                        row_heights=[0.4, 0.2, 0.2, 0.2] ) # Equal row heights ) # All subplots are 'xy' type by default)

    annotations = []
    for i, title in enumerate(subplot_titles_):
        row = (i // col_num) + 1
        col = (i % col_num) + 1
        y_position = 1 - (row - 1) / row_num + 0.02 if row == 1 else 1 - (row - 1) / row_num - 0.02

        print(row, col)
        annotations.append(dict(
            x= col / col_num - 1 / (2 * col_num),
            y = y_position,
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
        height=1200,
        width=1000,
        showlegend=True,
        annotations = annotations,
        yaxis1=dict(
            tickvals=[1e-1, 1, 10, 100, 1000, 10000, 100000],  
            ticktext=["0", "1", "10", "100", "1k", "10k", "100k"],  # Label 1e-1 as "0"
            type='log',  
            autorange=False,
            range=[-1, 5]  # adjust the range off of tickvals
        ),
        # ),
        yaxis4=dict(
            tickvals=[1e-2, 1e-1, 1, 10, 100],  # Start closer to 1
            ticktext=["0", "0.1", "1", "10", '100'],  # Label 1e-2 as "0"
            type='log',  
            autorange=False,
            range=[-2, 2]  # adjust the range off of tickvals
        )
    )

    for i, event in enumerate(event_type):
        vid_features =  video_metadata.filter(pl.col('event_type') == event)

        # Loop through columns and add a box plot for each
        for idx, column in enumerate(vid_feature_columns):
            row = (idx // col_num) + 1  # Calculate row number
            col = (idx % col_num) + 1   # Calculate column number
            # Add trace to the subplot
            # plotting the circles
            if row == 2 and col == 1:
                # live_values_df=vid_features.filter(pl.col('region') == 'US')['is_footage'].value_counts()
                # live_values_df = live_values_df.to_pandas()
                # live_values_df['is_footage'] = live_values_df['is_footage'].map({False: 'Not Footage', True: 'Footage'})
                # fig.add_trace(
                #     go.Pie(labels=live_values_df['is_footage'], 
                #         values=live_values_df['count'], 
                #         showlegend=True),
                #     row=row,
                #     col=col,
                # )
                # # print("circle plot for eu")
                # live_values_df=vid_features.filter(pl.col('region') == 'Europe')['is_footage'].value_counts()
                # live_values_df = live_values_df.to_pandas()
                # live_values_df['is_footage'] = live_values_df['is_footage'].map({False: 'Not Footage', True: 'Footage'})
                # fig.add_trace(
                #     go.Pie(labels=live_values_df['is_footage'], 
                #         values=live_values_df['count'], 
                #         showlegend=True),
                #     row=row,
                #     col=col+1,
                # )
                # # print("circle plot for asia")
                # live_values_df=vid_features.filter(pl.col('region') == 'Asia')['is_footage'].value_counts()
                # live_values_df = live_values_df.to_pandas()
                # live_values_df['is_footage'] = live_values_df['is_footage'].map({False: 'Not Footage', True: 'Footage'})
                # fig.add_trace(
                #     go.Pie(labels=live_values_df['is_footage'], 
                #         values=live_values_df['count'], 
                #         showlegend=True),
                #     row=row,
                #     col=col+2,
                # )
                fig = add_pie_chart_for_metric(fig, vid_features, 'is_footage', 'Footage', geographical_location, 'region', 2, 1)
                fig = add_pie_chart_for_metric(fig, vid_features, '_update_', 'Update', geographical_location, 'region', 3, 1)
                fig = add_pie_chart_for_metric(fig, vid_features, '_breaking_', 'Breaking', geographical_location, 'region', 4, 1)
            elif row == 3 and col == 1:
                metric_name = 'Update'
                for i,metric in enumerate(geographical_location):
                    live_values_df=video_metadata.filter(pl.col('region') == metric)['_update_'].value_counts()
                    live_values_df = live_values_df.to_pandas()
                    live_values_df['_update_'] = live_values_df['_update_'].map({False: f'Not {metric_name}', True: f'{metric_name}'})
                    
                    fig.add_trace(
                        go.Pie(labels=live_values_df['_update_'], 
                            values=live_values_df['count'], 
                            showlegend=True),
                        row=row,
                        col=column+i,
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
                            jitter=0.3,
                            pointpos=-2.0,
                            # width=0.001,
                            showlegend=False,
                            boxpoints='all' # suspectedoutliers, all, outliers
                            # boxmean='sd'  # Show mean and standard deviation
                        ),
                    row=row,
                    col=col,
                )
        label_name = event_type[i].capitalize()
        buttons.append(dict(
            args=[
                    {
                        'visible': [False]*i*num_subplots + [True]*num_subplots + [False]*(len(event_type)-i-1)*num_subplots
                    },
                    {
                        'title.text': f"Plot of Video Metrics for {label_name} Events"
                    }
                  ],
            label=label_name,
            method="update"
        ))

    initial_visibility = [True]*num_subplots + [False]*(len(event_type)-1)*num_subplots
    for i in range(len(fig.data)):
        fig.data[i].visible = initial_visibility[i]    

    label_name = event_type[0].capitalize()
    fig.update_layout(
        title=f"Box Plot of Video Metrics for {label_name} Events",
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
        print('saving')
        fig.write_html(save_path)


def plot_video_metrics_event_region(video_metadata: pl.DataFrame, save_path: str = None, show: bool = True):
    """
    Plot the video metrics discriminated by region, and plots the distribution based on events

      Args:
        df_channels (pl.DataFrame): df_channels_en unfiltered
        df_timeseries (pl.DataFrame): df_timeseries unfiltered
        save_path (str): path to save the plot as an HTML file
        show (bool): whether to display the plot
    """

    event_type = ['geopolitical', 'environmental']
    geographical_location = ['US', 'Europe', 'Asia']
    # plot settings
    vid_feature_columns = ['duration', 'channel_activity', 'subjectivity', 'capitalisation_ratio', 'is_footage']
    num_subplots = 4*3+3
    row_num = 2
    col_num = 4 
    subplot_titles_ = ['Duration (sec)', 'Channel Activity', 'Subjectivity', 
                 'Capitalization Ratio', ' ', 
                 'Is Footage %:<br>Geopolitical', 'Is Footage %:<br>Environmental']

    buttons = []
    # separated by event type
    fig = make_subplots(rows=row_num, cols=col_num, 
                        subplot_titles=subplot_titles_, 
                        horizontal_spacing = 0.1, vertical_spacing=0.18,
                        specs=[[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}, {'type': 'box'}],  
                                [{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}, None]],
                        column_widths=[0.25, 0.25, 0.25, 0.25,],  
                        row_heights=[0.5, 0.5] ) 

    annotations = []
    for i, title in enumerate(subplot_titles_):
        row = (i // col_num) + 1
        col = (i % col_num) + 1
        annotations.append(dict(
            # x= col / col_num - 1 / (2 * col_num),
            y = 1 - (row - 1) / row_num + 0.02 if row == 1 else 1 - (row - 1) / row_num - 0.13,
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
        height=800,
        width=800,
        showlegend=True,
        annotations = annotations,
        yaxis1=dict(
            tickvals=[1e-1, 1, 10, 100, 1000, 10000, 100000],  
            ticktext=["0", "1", "10", "100", "1k", "10k", "100k"],  
            type='log',  
            autorange=False,
            range=[-1, 5]  # Adjust the range accordingly
        ),
        # ),
        yaxis4=dict(
            tickvals=[1e-2, 1e-1, 1, 10, 100],  
            ticktext=["0", "0.1", "1", "10", '100'], 
            type='log',  # Logarithmic scale
            autorange=False,
            range=[-2, 2]  # Adjust the range accordingly
        )
    )
    # print(video_metadata.columns)
    # ################### INSERTION OF PLOT BASED ON REGION #################
    num_subplots_region_category = 4*2+2
    for j, region in enumerate(geographical_location):

        vid_features = video_metadata.filter(pl.col('region') == region)

        # Loop through columns and add a box plot for each
        for idx, column in enumerate(vid_feature_columns):
            row = (idx // col_num) + 1  # Calculate row number
            col = (idx % col_num) + 1   # Calculate column number
            # Add trace to the subplot
            # plotting the circles
            if row == 2 and col == 1:
                live_values_df=vid_features.filter(pl.col('event_type') == 'environmental')['is_footage'].value_counts()
                live_values_df = live_values_df.to_pandas()
                print(live_values_df)
                live_values_df['is_footage'] = live_values_df['is_footage'].map({False: 'Not Footage', True: 'Footage'})
                fig.add_trace(
                    go.Pie(labels=live_values_df['is_footage'], 
                        values=live_values_df['count'], 
                        showlegend=True),
                    row=row,
                    col=col+1,
                )
                # print("circle plot for asia")
                live_values_df=vid_features.filter(pl.col('event_type') == 'geopolitical')['is_footage'].value_counts()
                live_values_df = live_values_df.to_pandas()
                live_values_df['is_footage'] = live_values_df['is_footage'].map({False: 'Not Footage', True: 'Footage'})
                fig.add_trace(
                    go.Pie(labels=live_values_df['is_footage'], 
                        values=live_values_df['count'], 
                        showlegend=True),
                    row=row,
                    col=col+2,
                )
            else:
                # plotting box plots that keeps the events the same and compares across regions
                for event in event_type:
                    event_features = vid_features.filter(pl.col('event_type') == event)[column].to_list()
                    # print(event_features)
                    fig.add_trace(
                        go.Box(
                            y=event_features,
                            name=f"{event}",
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
        args=[
            {
                'visible': [False]*j*num_subplots_region_category + [True]*num_subplots_region_category + [False]*(len(geographical_location)-j-1)*num_subplots_region_category,
            },
            {
                'title.text': f"Plot of Video Metrics for {region} Events"
            }
        ],
        label=geographical_location[j],
        method="update"
        ))
    # Initial visibility for country
    initial_visibility = [True]*num_subplots_region_category + [False]*(len(geographical_location)-1)*num_subplots_region_category #+ [False] * len(geographical_location)*num_subplots_region_category
    print("iv: ", len(initial_visibility))
    for i in range(len(fig.data)):
        fig.data[i].visible = initial_visibility[i]  

    print("here")
    fig.update_layout(
        title=f"Plot of Video Metrics for US Events",
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

    fig.show(scrollZoom=False)
    if save_path:
        print('saving')
        fig.write_html(save_path)

def plot_video_metrics_response_event_type(comment_metadata: pl.DataFrame, save_path: str = None, show: bool = True):
    """
    Plot of metrics distinguished by event, and plots the response metrics based on region

      Args:
        df_channels (pl.DataFrame): df_channels_en unfiltered
        df_timeseries (pl.DataFrame): df_timeseries unfiltered
        save_path (str): path to save the plot as an HTML file
        show (bool): whether to display the plot
    """

    event_type = ['geopolitical', 'environmental']
    geographical_location = ['US', 'Europe', 'Asia']
    vid_feature_columns = ['view_count', 'total_comments', 'likes/comment', 'likes-dislikes/views']
    subplot_titles_ = ['Views', 'Total Comments', 'Likes per Comment', '(Likes-Dislikes) <br> Divided by Views']
    num_subplots = 4*3
    row_num = 2
    col_num = 2

    buttons = []
    fig = make_subplots(rows=row_num, cols=col_num, 
                        subplot_titles=subplot_titles_, 
                        horizontal_spacing = 0.1, vertical_spacing=0.15,
                        specs=[[{'type': 'box'}, {'type': 'box'}],  
                                [{'type': 'box'}, {'type': 'box'}]],
                        column_widths=[0.5, 0.5],  
                        row_heights=[0.5, 0.5] ) 

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
        height=800,
        width=800,
        showlegend=True,
        annotations = annotations,
        xaxis=dict(showticklabels=True),  
        xaxis2=dict(showticklabels=True),  
        xaxis3=dict(showticklabels=True),  
        xaxis4=dict(showticklabels=True), 
       # xaxis5=dict(showticklabels=True), 
        yaxis1=dict(
            tickvals=[1e-1,  1,  10,  100,   1000, 10000, 100000, 1000000, 10000000, 100000000],  # Start closer to 1
            ticktext=["0", "1", "10", "100", "1k", "10k", "100k", "1M", "10M", "100M"],  # Label 1e-2 as "0"
            type='log',  # Logarithmic scale
            autorange=False,
            range=[-1, 8]  # Adjust the range accordingly
        ),
        #  yaxis2=dict(
        #     tickvals=[1e-1,  1,  10,  100,   1000, 10000, 100000],  # Start closer to 1
        #     ticktext=["0", "1", "10", "100", "1k", "10k", "100k"],  # Label 1e-2 as "0"
        #     type='log',  # Logarithmic scale
        #     autorange=False,
        #     range=[-1, 5]  # Adjust the range accordingly
        # ),
        # # ),
        # yaxis4=dict(
        #     tickvals=[1e-2, 1e-1, 1, 10, 100],  # Start closer to 1
        #     ticktext=["0", "0.1", "1", "10", '100'],  # Label 1e-2 as "0"
        #     type='log',  # Logarithmic scale
        #     autorange=False,
        #     range=[-2, 2]  # Adjust the range accordingly
        # )
    )

    for i, event in enumerate(event_type):
        event_metadata =  comment_metadata.filter(pl.col('event_type') == event)

        # Loop through columns and add a box plot for each
        for idx, column in enumerate(vid_feature_columns):
            row = (idx // col_num) + 1  # Calculate row number
            col = (idx % col_num) + 1   # Calculate column number
            # Add trace to the subplot
            # plotting box plots comparing geographical location
            for region in geographical_location:
                # print(vid_features)
                # print(region)
                region_features = event_metadata.filter(pl.col('region') == region)[column].to_list()
                # print(region_features)
                fig.add_trace(
                    go.Box(
                        y=region_features,
                        name=f"{region}",
                        jitter=0.3,
                        pointpos=-2.0,
                        # width=0.001,
                        showlegend=False,
                        boxpoints='all' # suspectedoutliers, all, outliers
                        # boxmean='sd'  # Show mean and standard deviation
                    ),
                row=row,
                col=col,
            ) 
        event_label = event.capitalize()
        buttons.append(dict(
            args=[
                {
                    'visible': [False]*i*num_subplots + [True]*num_subplots + [False]*(len(event_type)-i-1)*num_subplots
                },
                {
                    'title.text': f"Plot of Response Metrics for {event_label} Events"
                }
                ],
            label=event_label,
            method="update"
        ))

    # adding visualization based off of region

    initial_visibility = [True]*num_subplots + [False]*(len(event_type)-1)*num_subplots
    for i in range(len(fig.data)):
        fig.data[i].visible = initial_visibility[i]    

    event_label = event_type[0].capitalize()
    fig.update_layout(
        title=f"Plot of Response Metrics for {event_label} Events",
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


def plot_video_metrics_response_region(comment_metadata: pl.DataFrame, save_path: str = None, show: bool = True):
    """
    Plot the video response metrics divided by regions, and plots the response metrics based on event types

      Args:
        df_channels (pl.DataFrame): df_channels_en unfiltered
        df_timeseries (pl.DataFrame): df_timeseries unfiltered
        save_path (str): path to save the plot as an HTML file
        show (bool): whether to display the plot
    """

    event_type = ['geopolitical', 'environmental']
    geographical_location = ['US', 'Europe', 'Asia']
    # plot settings
    vid_feature_columns = ['view_count', 'total_comments', 'likes/comment', 'likes-dislikes/views']
    subplot_titles_ = ['Views', 'Total Comments', 'Likes per Comment', '(Likes-Dislikes) <br> Divided by Views']
    num_subplots = 4*2
    row_num = 2
    col_num = 2

    buttons = []
    # separated by event type
    fig = make_subplots(rows=row_num, cols=col_num, 
                        subplot_titles=subplot_titles_, 
                        horizontal_spacing = 0.1, vertical_spacing=0.15,
                        specs=[[{'type': 'box'}, {'type': 'box'}],  
                                [{'type': 'box'}, {'type': 'box'}]],
                        column_widths=[0.5, 0.5],  
                        row_heights=[0.5, 0.5] ) 

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
        height=800,
        width=800,
        showlegend=True,
        annotations = annotations,
        xaxis=dict(showticklabels=True),  
        xaxis2=dict(showticklabels=True),  
        xaxis3=dict(showticklabels=True),  
        xaxis4=dict(showticklabels=True),
       # xaxis5=dict(showticklabels=True),  
        yaxis1=dict(
            tickvals=[1e-1,  1,  10,  100,   1000, 10000, 100000, 1000000, 10000000, 100000000],  # Start closer to 1
            ticktext=["0", "1", "10", "100", "1k", "10k", "100k", "1M", "10M", "100M"],  # Label 1e-2 as "0"
            type='log',  # Logarithmic scale
            autorange=False,
            range=[-1, 8]  # Adjust the range accordingly
        ),
        # # ),
        # yaxis4=dict(
        #     tickvals=[1e-2, 1e-1, 1, 10, 100],  # Start closer to 1
        #     ticktext=["0", "0.1", "1", "10", '100'],  # Label 1e-2 as "0"
        #     type='log',  # Logarithmic scale
        #     autorange=False,
        #     range=[-2, 2]  # Adjust the range accordingly
        # )
    )

    for i, region in enumerate(geographical_location):
        region_metadata =  comment_metadata.filter(pl.col('region') == region)

        # Loop through columns and add a box plot for each
        for idx, column in enumerate(vid_feature_columns):
            row = (idx // col_num) + 1  # Calculate row number
            col = (idx % col_num) + 1   # Calculate column number
            # Add trace to the subplot
            # plotting box plots comparing geographical location
            for event in event_type:
                # print(vid_features)
                # print(region)
                event_features = region_metadata.filter(pl.col('event_type') == event)[column].to_list()
                # print(region_features)
                fig.add_trace(
                    go.Box(
                        y=event_features,
                        name=f"{event}",
                        jitter=0.3,
                        pointpos=-2.0,
                        # width=0.001,
                        showlegend=False,
                        boxpoints='all', # suspectedoutliers, all, outliers
                        # boxmean='sd'  # Show mean and standard deviation
                    ),
                    # go.Violin
                    #     (
                    #         x=[event] * len(event_features),
                    #         y = event_features,
                    #         name = f"{event}",
                    #         box_visible=True,
                    #         meanline_visible=True,
                    #         showlegend=False
                    #     ),
                row=row,
                col=col,
            ) 
        # region_label = region.capitalize()
        buttons.append(dict(
            args=[
                {
                    'visible': [False]*i*num_subplots + [True]*num_subplots + [False]*(len(geographical_location)-i-1)*num_subplots
                },
                {
                    'title.text': f"Plot of Response Metrics for {region} Events"
                }
                ],
            label=region,
            method="update"
        ))

    # adding visualization based off of region
    initial_visibility = [True]*num_subplots + [False]*(len(geographical_location)-1)*num_subplots
    for i in range(len(fig.data)):
        fig.data[i].visible = initial_visibility[i]    

    fig.update_layout(
        title=f"Plot of Response Metrics for {geographical_location[0]} Events",
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

# def plot_timeseries(metadata, ):
    

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