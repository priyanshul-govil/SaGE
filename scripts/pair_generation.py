import argparse
import os
from itertools import combinations
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process input and output file paths.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()

    # Get the absolute paths for input and output files
    input_file_path = os.path.abspath(args.input_file)
    output_file_path = os.path.abspath(args.output_file)

    # Assuming df is your original DataFrame with the 'rot' column
    df = pd.read_csv(input_file_path)

    # Group the DataFrame by 'question_id' and get all possible pairs of rows within each group
    pairs = [list(combinations(group.iterrows(), 2))
             for _, group in df.groupby('question_id')]

    # Flatten the list of pairs
    pairs = [pair for group in pairs for pair in group]

    # Create a new DataFrame with the specified columns
    new_df = pd.DataFrame([
        {
            'question_id': pair[0][1]['question_id'],
            'paraphrased_1': pair[0][1]['paraphrased'],
            'model_output_1': pair[0][1]['model_output'],
            'rot_1': pair[0][1]['rot'],
            'paraphrased_2': pair[1][1]['paraphrased'],
            'model_output_2': pair[1][1]['model_output'],
            'rot_2': pair[1][1]['rot'],
        }
        for pair in pairs
    ])

    new_df.head()

    new_df.to_csv(output_file_path, index=False)
