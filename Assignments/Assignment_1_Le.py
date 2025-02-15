"""
Machine Learning Fundamentals, Spring 2025
Assignment 1
Author: Thinh Le
Date: January 27, 2025
"""

"""
Write a (or modify an existing) Python script to read in the Palmer Penguin 
dataset and explore it. Document each step of the code to demonstrate you 
understand what each block of code does. 
At a minimum, document:
    - What columns are in the dataset?
    - What penguins are included?
    - What islands, how many penguins in each group?
    - Any outliers among the quantitative data, and any missing data?
"""

"""
General notes:
    - The functions I used were documented on Pandas website: 
      https://pandas.pydata.org/docs/reference/index.html
"""

# Import libraries
import pandas as pd

# Read the Palmer Penguin cleaned dataset
# Note: place the dataset in the same folder with the python script.
penguins_df = pd.read_csv('penguins_clean.csv')


# Get an overview of data
# Note: Result is an DataFrame class from pandas.
print('\nGet an overview of data:')
print(penguins_df.info())


# Get dataset dimension
print('\nGet dataset dimension:')
print(penguins_df.shape)
# Result: (344, 8) - This dataset has 344 rows of data and 8 columns.


# Get the first 5 rows
# Note: If the number of rows, n, is not provided, 5 is the default value.
print('\nGet the first 5 rows:')
print(penguins_df.head())


# Get the last 5 rows
# Note: If the number of rows, n, is not provided, 5 is the default value.
print('\nGet the last 5 rows:')
print(penguins_df.tail())


# Get the columns names
columns = penguins_df.columns
# Notes:
#   - Result is an Index class from pandas.
#   - Result can be converted into Python list data type using tolist() method.
print('\nGet the columns names:')
print(columns.tolist())


# Get unique penguins
# 1: Get only species column
species = penguins_df['species']
# 2: Get unique species from the species column
unique_species = species.unique()
print('\nGet unique penguins:')
print(unique_species)


# Get unique islands
# 1: Get only island column
islands = penguins_df['island']
# 2: Get unique islands from the island column
unique_islands = islands.unique()
print('\nGet unique islands:')
print(unique_islands)


# Count the number of penguins in each island
# Version 1: Use groupby() function
#   1: Group data by island column
island_groups_v1 = penguins_df.groupby('island')
#   2: Count the number of penguins by each island
number_of_penguins_by_island = island_groups_v1['species'].count()
# Note: I select only the species column to display the count
print('\nThe number of penguins by each island (version 1):')
print(number_of_penguins_by_island)

# Version 2: Use groupby() function with agg() to apply a custom name for the
# count column. (https://www.geeksforgeeks.org/python-pandas-dataframe-groupby/)
#   1: Group data by island column, with as_index=False
island_groups_v2 = penguins_df.groupby('island', as_index=False)
#   Note:  as_index=False treats the grouped column (island) as a regular column
#   2: Use a new column name for the count column
number_of_penguins_by_island = island_groups_v2.agg(
    number_of_penguins=('species', 'count')
)
# Note: I select only the species column to display the count, but use
# 'number_of_penguins' as the new name
print('\nThe number of penguins in each island (version 2):')
print(number_of_penguins_by_island)


# Find outliers among the quantitative data
print('\nFind outliers among the quantitative data')

# 1: Create a list of quantitative columns
quantitative_column_names = [
    'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
    'body_mass_g', 'year'
]

# 2: Count the number of NAs in each column
print('\nCount the number of NAs in each column')
for column_name in quantitative_column_names:
    # Check for number of NAs in each column
    # (https://www.geeksforgeeks.org/check-for-nan-in-pandas-dataframe/)
    # 2.1: Check each row in the column if it has NA, this will set the value as
    # True if it is NA, and false otherwise using isna()
    column_na_check = penguins_df[column_name].isna()
    # 2.2: Check if at least a True value in the column
    is_column_has_na = column_na_check.any()

    if is_column_has_na :
        # 2.3: If column has NAs, print the number of NAs
        number_of_nas = column_na_check.sum()
        print(f"Column '{column_name}' has {number_of_nas} rows having NAs")
        # f-strings: https://www.geeksforgeeks.org/python-string-interpolation/

# 3: Drop NAs from the dataframe
penguins_df.dropna(inplace=True)
# inplace=True: modify the penguins_df rather than creating a new variable to
# store the result
print('\nNAs dropped')

# 4: View summary of each column if there is any outlier
print('\nView summary of each column if there is any outlier')

for column_name in quantitative_column_names:
    # 4.1 View summary of all columns except the year column
    if column_name != 'year':
        print(penguins_df[column_name].describe())
        print() # print an empty line for better reading
    # 4.2 If it is the year column, I just want to find the unique years
    else:
        unique_years = penguins_df[column_name].unique()
        print('Unique years:')
        print(unique_years)

# Note: The results are nice, no additional outliers detected
