# # load yt metadata in chunks and filter for videos contained in filtered_df_ch
# # df.filter(pl.col("categories") == "News & Politics") to filter for categories of videos instead
import polars as pl
import pandas as pd
import time 
import os

# iterate through the comment files 
tsv_gz_files = []
directory = 'comments_line_separated'
for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".gz"):
            tsv_gz_files.append(os.path.join(root, file))
            
print(tsv_gz_files)
video_metadata_df = pl.read_csv("data/final_yt_metadata_helper.csv")
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