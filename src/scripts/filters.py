import polars as pl
import pandas as pd

def filter_df(df: pl.DataFrame, column_name: str, value: str, cmpstr : str = "==") -> pl.DataFrame:
    """
    Filter a DataFrame based on a column and a value.
    
    Args:
        df: The DataFrame to filter.
        column_name: The name of the column to filter.
        value: The value to filter on.
        cmpstr: The comparison string. Default is "=".
        
    Returns:
        The filtered DataFrame.
    """
    assert column_name in df.columns, \
        f"Column '{column_name}' not found in DataFrame. Provide one of {df.columns}."
    
    if cmpstr == "==":
        return df.filter(pl.col(column_name) == value)
    elif cmpstr == "!=":
        return df.filter(pl.col(column_name) != value)
    elif cmpstr == "<":
        return df.filter(pl.col(column_name) < value)
    elif cmpstr == "<=":
        return df.filter(pl.col(column_name) <= value)
    elif cmpstr == ">":
        return df.filter(pl.col(column_name) > value)
    elif cmpstr == ">=":
        return df.filter(pl.col(column_name) >= value)
    else:
        raise ValueError("Invalid comparison string. Use one of '==', '!=', '<', '<=', '>', '>='.")
    
    
def filter_df_isin(df: pl.DataFrame, column_name: str, values: list|pl.series.series.Series) -> pl.DataFrame:
    """
    Filter a DataFrame based on a column and a list of values.
    
    Args:
        df: The DataFrame to filter.
        column_name: The name of the column to filter.
        values: The list of values to filter on.
        
    Returns:
        The filtered DataFrame.
    """
    assert column_name in df.columns, \
        f"Column '{column_name}' not found in DataFrame. Provide one of {df.columns}."
    
    
    return df.filter(pl.col(column_name).is_in(values))

    

def df_filter_csv_batched(input_path: str, output_path: str, column_name: str, 
                          values: type, filter_method: str,
                          sep_in: str = "\t", sep_out: str = "\t", batch_size=5000):
    """
    Filter a DataFrame in batches.
    
    Args:
        input_path: the path to input csv file
        output_path: the path to the save the filtered output csv file
        column_name: the name of the column to apply the filter
        values: the value(s) to filter the column
        filter_method: The filtring operation. One of 'is_in', '==', '!=', '<', '<=', 
                       '>', '>='.
        sep_in: separator used in the input csv file. Default "\t".
        sep_out: separator used in the output csv file. Default "\t".
        batch_size: Sizer per batch. Default 5000
    """

    reader = pl.read_csv_batched(input_path, separator=sep_in, batch_size=batch_size)

    batches = reader.next_batches(5)  
    i = 0
    isin = (filter_method == "is_in")
    while batches:
        for df in batches:
            if isin:
                filtered_df = filter_df_isin(df, column_name, values)
            else:
                filtered_df = filter_df(df, column_name, values, cmpstr=filter_method)

            if i == 0:
                filtered_df.write_csv(output_path, include_header=True, separator=sep_out)
            else:
                with open(output_path, "a") as fh:
                    fh.write(filtered_df.write_csv(file=None, include_header=False, 
                                                   separator=sep_out))
            i = i+1
            print(f"processing batch {i} ...\r", end='')
        batches = reader.next_batches(5)

def df_filter_jsonl_batched(input_path: str, output_path: str, column_name: str, 
                          values: type, sep: str = "\t", batch_size=5000):
    """
    Filter a jsonl file in batches using pandas and saves it in a csv file.
    
    Args:
        input_path: the path to input csv file
        output_path: the path to the save the filtered output csv file
        column_name: the name of the column to apply the filter
        values: the value(s) to filter the column
        sep: separator used in the csv file. Default "\t".
        batch_size: Sizer per batch. Default 5000
    """
    chunks = pd.read_json(input_path, lines=True, chunksize = batch_size)
    for i, c in enumerate(chunks):
        c = c[c[column_name].isin(values)]
        if i == 0:
            print(c)
            c.to_csv(output_path, header=True, index=False, sep=sep)
        else: 
            with open(output_path, "a") as fh:
                fh.write(c.to_csv(path_or_buf=None, header=False, index=False, 
                                  sep = sep))
        print(f"processing batch {i} \r", end="")