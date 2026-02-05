import pandas as pd
import ast
import argparse
import os

def add_label_to_extra_info(row, label_value):
    """Add or update the 'labeled' field in extra_info"""
    extra = row['extra_info']
    
    if isinstance(extra, str):
        try:
            info_dict = ast.literal_eval(extra)
            if not isinstance(info_dict, dict):
                info_dict = {}
        except (ValueError, SyntaxError):
            info_dict = {}
        info_dict['labeled'] = bool(label_value)
        return str(info_dict)
    elif isinstance(extra, dict):
        new_dict = extra.copy()
        new_dict['labeled'] = bool(label_value)
        return new_dict
    else:
        return {'labeled': bool(label_value)}

def update_extra_info_index(row, new_index):
    """Update the 'index' field in extra_info, preserving the original format (str or dict)"""
    extra = row['extra_info']
    
    if isinstance(extra, str):
        try:
            info_dict = ast.literal_eval(extra)
            if not isinstance(info_dict, dict):
                info_dict = {}
        except (ValueError, SyntaxError):
            info_dict = {}
        info_dict['index'] = new_index
        return str(info_dict)
    elif isinstance(extra, dict):
        new_dict = extra.copy()
        new_dict['index'] = new_index
        return new_dict
    else:
        return {'index': new_index}

def main(pos_file, neg_file, output_file):
    # Read two Parquet files
    print(f"Reading positive samples file: {pos_file}")
    df_pos = pd.read_parquet(pos_file)
    print(f"Reading negative samples file: {neg_file}")
    df_neg = pd.read_parquet(neg_file)

    # Add label field
    print("Adding label=True to positive samples...")
    df_pos['extra_info'] = [
        add_label_to_extra_info(row, True) for _, row in df_pos.iterrows()
    ]
    
    print("Adding label=False to negative samples...")
    df_neg['extra_info'] = [
        add_label_to_extra_info(row, False) for _, row in df_neg.iterrows()
    ]

    # Concatenate DataFrames
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    print(f"Merging completed. Total number of samples: {len(df_combined)}")

    # Update index to sequential integers
    print("Updating 'index' in extra_info to sequential indices...")
    df_combined['extra_info'] = [
        update_extra_info_index(row, i) for i, row in df_combined.iterrows()
    ]

    # Save result
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df_combined.to_parquet(output_file, index=False)
    print(f"Saved merged file to: {output_file}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Merge two Parquet files, label samples (positive=True, negative=False), and reset the index."
    # )
    # parser.add_argument("pos_file", help="Path to the positive samples Parquet file (label=True)")
    # parser.add_argument("neg_file", help="Path to the negative samples Parquet file (label=False)")
    # parser.add_argument("output_file", help="Path to save the merged Parquet file")

    # args = parser.parse_args()

    main(
        pos_file='/data/ID_domain/openr1_1024_fixed.parquet',
        neg_file='/data/ID_domain/openr1_1024_1024_fixed.parquet',
        output_file='/data/ID_data/processed/id_l_1k_u_1k_fixed.parquet'
    )