import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

def plot_frequency_public_metrics(regions=[], event_list=[], response_metrics = [], public_response_df=None):
    '''
        Note: you can only plot either region or event

        Inputs: 
            event_list: type of event you want to plot 
                - Natural disasters, geopolitical events
            region: region for event you want to plot
                - US, Europe, Asia
            response_metrics: which response metrics to plot
                - 1. like_count, 2. dislike_count, 
                  3. view_count, 4. num_comments, 
                  5. num_comments_likes 6. num_comments_replies
            public_response_df: the dataframe containing responses to plot
    ''' 
    # plotting metrics based on event types
    if len(regions) != 0:
        print(len(regions))
        # Create a grid of subplots (rows = categories, cols = metrics)
        fig, axes = plt.subplots(len(regions), len(response_metrics), figsize=(20, 15), constrained_layout=True)
        if len(axes) > 1:
            axes = axes.reshape(len(regions), len(response_metrics))
        colors = sns.color_palette("tab10", len(response_metrics))
        filter_df = public_response_df.filter((pl.col("Region").is_in(regions))).select(['Region'] + response_metrics).to_pandas()
        # sum_df = filter_df.groupby('Region').sum()

        # print(sum_df.head)
        # Loop through each category and metric to create individual histograms
        for i, region in enumerate(regions):
            for j, metric in enumerate(response_metrics):
                 # Filter the DataFrame for the specific region
                filtered_data = filter_df[filter_df['Region'] == region]

                # Plot the histogram on the specific axis
                sns.histplot(data=filtered_data, x=metric, bins=10, color=colors[j], ax=axes[i, j], kde=False)

                # Set the title for each subplot
                # axes[i, j].set_title(f'{region} - {metric}')
                axes[i, j].set_title(f"{region} - {metric}")
                axes[i, j].set_xlabel(f"{metric}")
                axes[i, j].set_ylabel("Frequency")
                # for bar in ax.patches:
                #     bar.set_edgecolor('black') 

        # Add a global title
        plt.suptitle("Frequency Histograms for Each Metric by Region", fontsize=20)
        plt.show()

    elif len(event_list) != 0:
        # Create a grid of subplots (rows = categories, cols = metrics)
        fig, axes = plt.subplots(len(event_list), len(response_metrics), figsize=(20, 15), constrained_layout=True)
        if len(axes) > 1:
            axes = axes.reshape(len(event_list), len(response_metrics))

        colors = sns.color_palette("flare", n_colors=len(response_metrics))
        filter_df = public_response_df.filter((pl.col("Event").is_in(event_list))).select(['Event'] + response_metrics).to_pandas()
        # sum_df = filter_df.groupby('Event').sum()
        print(filter_df)
        # Loop through each category and metric to create individual histograms
        for i, event in enumerate(event_list):
            for j, metric in enumerate(response_metrics):
                filtered_data = filter_df[filter_df['Event'] == event]

                # Plot the histogram on the specific axis
                sns.histplot(data=filtered_data, x=metric, bins=10, color=colors[j], ax=axes[i, j], kde=False)

                axes[i, j].set_title(f"{event} - {metric}")
                axes[i, j].set_xlabel(f"{metric}")
                axes[i, j].set_ylabel("Frequency")

        # Add a global title
        plt.suptitle("Frequency Histograms for Each Metric by Event", fontsize=20)
        plt.show()




