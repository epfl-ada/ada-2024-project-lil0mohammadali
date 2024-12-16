import polars as pl
from ..scripts import filters
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
                        column_widths=[0.7, 0.3])
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