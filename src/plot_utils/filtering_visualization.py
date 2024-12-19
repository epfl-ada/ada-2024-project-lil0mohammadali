from plotly import graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_filtering_visualization(y: list, x: list, save_path: str = None, 
                                 show: bool = True):
    """
    Plot a funnel chart with the given data.
    
    Args:
        y (list): List of strings for the y-axis (name of filtering steps)
        x (list): List of numbers for the x-axis (number of items at each step)
        save_path (str): Path to save the plot as an HTML file
        show (bool): Whether to display the plot
    """
    assert len(y) == len(x), "The length of the lists must be equal"
    colors = ["rgb(5,10,172)",
    "rgb(40,60,190)",
    "rgb(70,100,245)",
    "rgb(90,120,245)"]
    fig = go.Figure(go.Funnel(
        y = y,
        x = x,
        textposition = "inside",
        textinfo = "value+percent initial",
        opacity = 0.65,
        marker = {"color": colors},
        # connector = {"line": {"color": "royalblue", "dash": "dot", "width": 3}},)
        ))
    fig.update_layout(title_text="Number of Channels in Dataset per Filtering Step")
    if show:
        fig.show()
    if save_path:
        fig.write_html(save_path)


def sankey_plot_events(df: pd.DataFrame, save_path: str = None, show: bool = True):
    """
    Plot a Sankey diagram for the given DataFrame.
    
    Args:
        df (pl.DataFrame): DataFrame with columns 'region', 'event_type', 'event'
        save_path (str): Path to save the plot as an HTML file
        show (bool): Whether to display the plot
    """

    counts = df.groupby(['event_type', 'region', 'event']).size().reset_index(name='count')

    labels = ['All Videos from CoI']  # root label
    event_types = np.sort(df['event_type'].unique())
    event_locations = np.sort(df['region'].unique())
    event_names = list(counts['event'].unique())

    # Add labels for event_types, location and names
    labels.extend(event_types)  
    for event_type in event_types:
        labels.extend([f"{location} ({event_type})" for location in event_locations]) 
    labels.extend(event_names) 

    sources = []
    targets = []
    values = []

    link_colors = []
    node_colors = []

    blue = [px.colors.sequential.Blues[4], px.colors.sequential.Blues[1], 
            px.colors.sequential.Blues[3], px.colors.sequential.Blues[5]]
    red =  [px.colors.sequential.OrRd[4], px.colors.sequential.OrRd[1], 
            px.colors.sequential.OrRd[3], px.colors.sequential.OrRd[5]]
   

    for i, node in enumerate(labels):
        if i == 0:
            # grey
            node_colors.append("rgba(200,200,200,0.7)")
        elif i < len(event_types) + 1:
            if node == event_types[0]:
                node_colors.append(red[0])
            else:
                node_colors.append(blue[0])
          
    # "All Videos" -> event_type
    for event_type in event_types:
        sources.append(0) 
        targets.append(labels.index(event_type))
        values.append(counts[counts['event_type'] == event_type]["count"].sum())

        if event_type == event_types[0]:
            link_colors.append(red[0]) 
        else:
            link_colors.append(blue[0]) 

    # event_type --> event_location
    for event_type in event_types:
        for i, event_location in enumerate(event_locations):
            sources.append(labels.index(event_type))
            targets.append(labels.index(f"{event_location} ({event_type})")) 
            values.append(counts[(counts['event_type'] == event_type) & 
                                 (counts['region'] == event_location)]["count"].sum())
            if event_type == event_types[0]:
                link_colors.append(red[i+1]) 
                node_colors.append(red[i+1])
            else:
                link_colors.append(blue[i+1]) 
                node_colors.append(blue[i+1])
                
    # event_location -> event_name
    for event_type in event_types:
        for i, event_location in enumerate(event_locations):
            for event_name in event_names:
                count = counts[(counts['event_type'] == event_type) &
                            (counts['region'] == event_location) &
                            (counts['event'] == event_name)]['count'].sum()
                if count > 0:
                    sources.append(labels.index(f"{event_location} ({event_type})")) 
                    targets.append(labels.index(event_name)) 
                    values.append(count)

                    if event_type == event_types[0]:
                        node_colors.append(red[i+1])
                        link_colors.append(red[i+1]) 
                    else:
                        link_colors.append(blue[i+1]) 
                        node_colors.append(blue[i+1])

    # Step 4: Create the Sankey diagram
    sankey_fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        )
    ))

    sankey_fig.update_layout(title_text="Event Filtering", font_size=10)
    if show:
        sankey_fig.show()
    if save_path:
        sankey_fig.write_html(save_path)
