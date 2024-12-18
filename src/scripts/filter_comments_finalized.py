# # load yt metadata in chunks and filter for videos contained in filtered_df_ch
# # df.filter(pl.col("categories") == "News & Politics") to filter for categories of videos instead
import polars as pl
import pandas as pd
import time 
import os
import shutil


def filter_relevant_comments(directory, video_metadata_file):
    # iterate through the comment files 
    tsv_gz_files = []
    # directory = 'comments_line_separated'
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                tsv_gz_files.append(os.path.join(root, file))
                
    print(tsv_gz_files)
    # video_metadata_df = pl.read_csv("data/final_yt_metadata_helper.csv")
    video_metadata_df = pl.read_csv(video_metadata_file)
    # Define the chunk size (number of rows per chunk)
    chunk_size = 100000
    NUM_ROWS = 400000000
    total_chunks = NUM_ROWS / chunk_size
    # print(video_metadata_df.shape)

    for i in range(len(tsv_gz_files)):
        file_path = tsv_gz_files[i]
        print("Currently processing: ", file_path)
        
        start_time = time.time()

        # create an iterator for processing the file in chunks, set header=None if part is not "aa"
        chunk_iterator = pd.read_csv(file_path, 
                                    sep='\t',  
                                    compression='gzip',  
                                    chunksize=chunk_size,
                                    header=None, 
                                    iterator=True)

        out_file = os.path.basename(tsv_gz_files[i])
        out_file = out_file[:-3]
        out_file = out_file + "_filtered.csv"
        print(out_file)
        
        for chunk_number, chunk in enumerate(chunk_iterator):
            pl_chunk = pl.from_pandas(chunk)
            
            filtered_chunk = pl_chunk.filter(pl.col("1").is_in(video_metadata_df["display_id"]))
            if chunk_number == 0:
                filtered_chunk.write_csv(out_file, include_header=True)
            else:
                with open(out_file, mode="a") as f:
                    filtered_chunk.write_csv(f, include_header=False)
            if chunk_number % 20 == 0:
                print(f"{chunk_number} / {total_chunks} processed" )
                
                
        # print time to process files
        end_time = time.time()
        elapsed_seconds = end_time - start_time

        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"Total processing time for file {os.path.basename(file_path)}: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

def combine_csv_files_polars(input_directory, output_file):
    combined_data = []

    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"): 
                file_path = os.path.join(root, file)
                print(f"Reading: {file_path}")
                df = pl.read_csv(file_path)
                combined_data.append(df)

    combined_df = pl.concat(combined_data)
        
    combined_df.write_csv(output_file)
    print(f"Combined CSV saved to: {output_file}")

def update_csv_header(input_file, output_file, replacement_line):
    from_file = open(input_file, 'r')
    to_file = open(output_file, 'w')

    replacement_line = 'author,video_id,likes,replies\n'
    from_file.readline() # and discard
    to_file.write(replacement_line)
    shutil.copyfileobj(from_file, to_file)   


def write_comments_statistics_csv(video_metadata_file, filtered_comments_file):

    video_metadata_df = pl.read_csv(video_metadata_file)

    # reader = pl.read_csv_batched("data/updated_news_videos_filtered_comments.csv", batch_size = 20000)  
    reader = pl.read_csv_batched(filtered_comments_file, batch_size = 20000)
    batches = reader.next_batches(20)  

    comment_stats_df = video_metadata_df.select('display_id')
    comment_stats_df = comment_stats_df.with_columns(pl.lit(0).alias('num_comments'))
    comment_stats_df = comment_stats_df.with_columns(pl.lit(0).alias('total_likes'))
    comment_stats_df = comment_stats_df.with_columns(pl.lit(0).alias('num_replies'))
    print(comment_stats_df.head)

    processed_batches = 0
    batch_size = 20000
    num_rows = 143000000
    total_batches = num_rows/batch_size

    while batches: 
        df_current_batches = pl.concat(batches)
        # print(type(df_current_batches))
        result_df = df_current_batches.group_by("video_id").agg([
            pl.col("likes").sum().alias("total_likes"),
            pl.col("replies").sum().alias("total_replies"),
            pl.len().alias("num_comments")
        ]) 
        # print(result_df.head)        
        # Update comment_stats_df with the aggregated data
        comment_stats_df = comment_stats_df.join(result_df, left_on="display_id", right_on="video_id", how="left").with_columns([
            (pl.col("num_comments") + pl.col("num_comments_right").fill_null(0)).alias("num_comments"),
            (pl.col("total_likes") + pl.col("total_likes_right").fill_null(0)).alias("total_likes"),
            (pl.col("num_replies") + pl.col("total_replies").fill_null(0)).alias("num_replies"),
            ]).select(["display_id", "num_comments", "total_likes", "num_replies"])  # Keep only necessary columns
        
        
        batches = reader.next_batches(20)
        processed_batches += 20
        print("Proportion of batches processed: ", processed_batches/total_batches)


    comment_stats_df.write_csv("../../data/comment_stats.csv")