import polars as pl

keywords = [
    ["footage"],
    # ["update"],
    # ["breaking"]
]

def title_contains(term):
    pattern = f"(?i){term}"  # case-insensitive
    # pattern = f"(?i)\\b{term}\\b"  # case-insensitive, match whole word
    return (pl.col("title").str.contains(pattern, literal=False))

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
    print("complex terms: ", complex_terms)
    for group in complex_terms:
        group_condition = pl.lit(True)
        for term in group:
            group_condition = (group_condition & title_contains(term))
        final_condition = final_condition | group_condition
    
    return final_condition

def add_video_live(df):
    final_condition = build_filter_condition(keywords)
    df = df.with_columns((final_condition).alias("is_footage"))
    return df
