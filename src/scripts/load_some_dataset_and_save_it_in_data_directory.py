import polars as pl

# Create a sample DataFrame
df = pl.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
})

# Write the DataFrame to a CSV file
df.write_csv("output.csv")

print("CSV file has been created successfully.")