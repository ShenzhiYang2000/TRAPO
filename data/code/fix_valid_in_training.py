import pandas as pd

# 1. Read the original Parquet file
input_path = "/data/valid_in_training.parquet"      # Replace with your input file path
output_path = "/data/valid_in_training_fixed.parquet"    # Replace with your desired output file path

df = pd.read_parquet(input_path)

# 2. Define the new system prompt content
new_content = r"Let's think step by step and output the final answer within \boxed{}."

# 3. Update the 'content' field of the first dictionary in the 'prompt' list for each sample
def update_first_prompt(prompts):
    prompts = prompts.copy()          # Avoid modifying the original list
    prompts[0] = prompts[0].copy()    # Avoid modifying the original dictionary
    prompts[0]['content'] = new_content
    return prompts

df['prompt'] = df['prompt'].apply(update_first_prompt)

# 4. Save to a new Parquet file
df.to_parquet(output_path, index=False)

print(f"✅ Modification completed and saved to: {output_path}")