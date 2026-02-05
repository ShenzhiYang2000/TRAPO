import pandas as pd
import pyarrow.parquet as pq

def sample_parquet_file(input_path, output_path, sample_size=512, random_state=None):
    """
    Randomly samples a specified number of rows from a Parquet file and saves them to a new file.

    Parameters:
    input_path: Path to the input Parquet file
    output_path: Path to the output Parquet file
    sample_size: Number of samples to draw; default is 512
    random_state: Random seed for reproducible results
    """
    try:
        # Load the entire Parquet file into a DataFrame
        df = pd.read_parquet(input_path)
        
        # If the file contains fewer rows than the requested sample size, use all rows
        if len(df) <= sample_size:
            sampled_df = df
            print(f"Warning: The file contains only {len(df)} rows, which is fewer than the requested {sample_size}.")
        else:
            # Randomly sample the specified number of rows
            sampled_df = df.sample(n=sample_size, random_state=random_state, replace=False)
        
        # Save the sampled data to a new Parquet file
        sampled_df.to_parquet(output_path, index=False)
        print(f"Successfully sampled {len(sampled_df)} rows and saved to {output_path}")
        
        return sampled_df
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None


if __name__ == "__main__":
    input_file = ""
    output_file = ""
    
    sampled_data = sample_parquet_file(input_file, output_file, sample_size=2048, random_state=42)