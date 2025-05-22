import pandas as pd

# Load TSV
df = pd.read_csv("train.tsv", sep="\t")

# Filter non-empty accent labels
accent_df = df[df["accents"].notnull() & (df["accents"] != "")]

# Show how many valid rows exist
print(f"Total rows with accents: {len(accent_df)} / {len(df)}")

# Show accent class distribution
print(accent_df["accents"].value_counts())

# Optional: Save this to a CSV for review
accent_df.to_csv("filtered_with_accents.csv", index=False)