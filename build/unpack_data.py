import os
import pandas as pd

def unpack_data(input_dir, output_file):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.
    """

    # Step 1: Initialize an empty list to store DataFrames
    dataframes=[]

    # Step 2: Loop over files in the input directory

    directory = os.listdir(input_dir)
    print("Fichiers trouvés :", directory)

    for file in directory:
        path = os.path.join(input_dir, file)
        if os.path.isfile(path): 
            print(f"Lecture du fichier : {path}")
            df = pd.read_csv(path, header=None, names=[
            'sequence', 'family_accession', 'sequence_name', 'aligned_sequence', 'family_id'
            ])
            dataframes.append(df)

        # Step 3: Check if the file is a CSV or matches a naming pattern

        # Step 4: Read the CSV file using pandas

        # Step 5: Append the DataFrame to the list

    # Step 6: Concatenate all DataFrames
    final_df= pd.concat(dataframes, ignore_index=True)

    # Step 7: Save the combined DataFrame to output_file
    final_df.to_csv(output_file, index=False)
    print(f"Fichier combiné sauvegardé à : {output_file}")


    pass


if __name__ == "__main__":
    print("Script démarré")

    import sys
    sys.argv = [
        "unpack_data.py", 
        "--input_dir", "./Data-Lakes-tp1-student/data/bronze/random_split/dev",
        "--output_file", "./Data-Lakes-tp1-student/data/bronze/random_split/combined_df.csv"
    ]

    import argparse
    parser = argparse.ArgumentParser(description="Unpack and combine protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output combined CSV file")
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file)
