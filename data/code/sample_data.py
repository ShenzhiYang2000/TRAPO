import pandas as pd
import pyarrow.parquet as pq

def sample_parquet_file(input_path, output_path, sample_size=512, random_state=None):
    """
    Randomly sample a specified number of rows from a Parquet file and save them to a new file.

    Parameters:
    input_path: Path to the input Parquet file.
    output_path: Path to the output Parquet file.
    sample_size: Number of samples to draw. Default is 512.
    random_state: Random seed for reproducibility.
    """
    try:
        # Read the entire Parquet file into a DataFrame
        df = pd.read_parquet(input_path)
        
        # If the file contains fewer rows than the requested sample size, use all rows
        if len(df) <= sample_size:
            sampled_df = df
            print(f"Warning: The input file contains only {len(df)} rows, which is fewer than the requested {sample_size}.")
        else:
            # Randomly sample the specified number of rows without replacement
            sampled_df = df.sample(n=sample_size, random_state=random_state, replace=False)
        
        # Save the sampled data to a new Parquet file
        sampled_df.to_parquet(output_path, index=False)
        print(f"Successfully sampled {len(sampled_df)} rows and saved to {output_path}")
        
        return sampled_df
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None

# Example usage
if __name__ == "__main__":
    input_file = "/data/valid.mmlu_pro.parquet"  # Replace with your input file path
    output_file = "/data/train.mmlu_pro-128.parquet"  # Replace with your desired output path
    
    # Perform sampling
    sampled_data = sample_parquet_file(input_file, output_file, sample_size=128, random_state=42)