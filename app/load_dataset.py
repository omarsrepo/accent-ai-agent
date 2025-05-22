from datasets import load_dataset

"""
#Downloads just ONE FILE named "en/c4-train.00001-of-01024.json.gz" 
#from the hugginface repository allenai/c4/en.
#c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz")

#Downloads the entire folder named "en" from the repository allenai/c4
#c4_subset = load_dataset("allenai/c4", data_dir="en")
"""

# Obtaining the english training split audio files from the mozilla common voice 17 repository on huggingface
train_split = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train")