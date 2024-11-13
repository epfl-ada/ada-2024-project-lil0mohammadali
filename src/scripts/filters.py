import polars as pl

def filter_df(df: pl.DataFrame, column_name: str, value: str, cmpstr : str = "=") -> pl.DataFrame:
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
    assert column_name in df.columns, / 
    f"Column '{column_name}' not found in DataFrame. Provide one of {df.columns}."
    
    assert isinstance(value), "Value must be a string."
    
    if cmpstr == "=":
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
        raise ValueError("Invalid comparison string. Use one of '=', '!=', '<', '<=', '>', '>='.")
    
    
def filter_df_isin(df: pl.DataFrame, column_name: str, values: list) -> pl.DataFrame:
    """
    Filter a DataFrame based on a column and a list of values.
    
    Args:
        df: The DataFrame to filter.
        column_name: The name of the column to filter.
        values: The list of values to filter on.
        
    Returns:
        The filtered DataFrame.
    """
    assert column_name in df.columns, / 
    f"Column '{column_name}' not found in DataFrame. Provide one of {df.columns}."
    
    assert isinstance(values, list), "Values must be a list."
    
    return df.filter(pl.col(column_name).isin(values))

def get_CoI(path_df_channel_en: str, output_path: str):
    """
 
    """
    df_channels = pl.read_csv(path)
    df_filtered = filter_df(df_channels, "category", "News & Politics")
    df_filtered.write_csv(output_path)
    
    
