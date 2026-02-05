import pyarrow.parquet as pq
import pandas as pd
import ast

def update_extra_info(extra_info, new_index):
    if isinstance(extra_info, str):
        try:
            info_dict = ast.literal_eval(extra_info)
            if isinstance(info_dict, dict):
                info_dict['index'] = new_index
                return str(info_dict)
        except (ValueError, SyntaxError):
            pass
        return extra_info
    elif isinstance(extra_info, dict):
        info_dict = extra_info.copy()
        info_dict['index'] = new_index
        return info_dict
    else:
        return extra_info

def process_parquet(
    input_path: str,
    output_path: str,
    start_row: int,
    num_rows: int,
    index_base: int = None
):

    if index_base is None:
        index_base = start_row

    table = pq.read_table(input_path)
    sliced_table = table.slice(start_row, num_rows)
    df = sliced_table.to_pandas()

    df['extra_info'] = [
        update_extra_info(val, index_base + i)
        for i, val in enumerate(df['extra_info'])
    ]

    df.to_parquet(output_path, index=False)
    print(f"Saved {num_rows} rows to '{output_path}' with index starting at {index_base}.")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Parquet file and update extra_info index.")
    parser.add_argument("input", help="Input Parquet file path")
    parser.add_argument("output", help="Output Parquet file path")
    parser.add_argument("--start", type=int, required=True, help="Start row index")
    parser.add_argument("--rows", type=int, required=True, help="Number of rows to read")
    parser.add_argument("--index-base", type=int, default=0, help="Base index for 'index' field (default: same as start)")

    args = parser.parse_args()
    process_parquet(
        input_path=args.input,
        output_path=args.output,
        start_row=args.start,
        num_rows=args.rows,
        index_base=args.index_base
    )