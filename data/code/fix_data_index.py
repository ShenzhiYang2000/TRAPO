import pandas as pd
import ast
 import json

# Read the parquet file
df = pd.read_parquet('/data/ID_domain/openr1_2048_2048.parquet')

# Method 1: If extra_info is a string representation of a dictionary
def update_extra_info(row, new_index):
    if isinstance(row['extra_info'], str):
        try:
            info_dict = ast.literal_eval(row['extra_info'])
            info_dict['index'] = new_index
            return str(info_dict)
        except:
            return row['extra_info']
    elif isinstance(row['extra_info'], dict):
        row['extra_info']['index'] = new_index
        return row['extra_info']
    else:
        return row['extra_info']

# Apply the function to update the 'index' field for each sample
df['extra_info'] = [update_extra_info(row, i) for i, row in df.iterrows()]

# Save the updated dataframe to a new parquet file
df.to_parquet('/data/ID_domain/openr1_2048_2048_fixed.parquet', index=False)
print("The 'index' field in extra_info has been updated to sequential indices.")