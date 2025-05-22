import pandas as pd
import os

# Load TSV
df = pd.read_csv("train.tsv", sep="\t")

# Step 1: Remove all rows where accents column is empty (Filtering out empty/null accents from the dataset)
accent_df = df[df["accents"].notnull() & (df["accents"] != "")]
print(f"Total rows with accents: {len(accent_df)} / {len(df)}")
print(accent_df["accents"].value_counts())
accent_df.to_csv("filtered_with_accents.csv", index=False)

# Step 2: Check which files exist
audio_folder = r"c:\Users\GAME\Desktop\en train dataset"
available_files = set(os.listdir(audio_folder))

# Step 3: Keep only rows where audio file exists
final_df = accent_df[accent_df["path"].isin(available_files)]
print(f"Rows with matching audio files: {len(final_df)}")

# We get a csv file with accents column populated and only contains rows for audio files we have readily available
final_df.to_csv("filtered_train.csv", index=False) 

