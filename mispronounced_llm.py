import os
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset
import json

class MispronouncedWordsDataset:
    def __init__(self, csv_file="mispronounced.csv"):
        self.csv_file = csv_file
        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["original"] + [f"mispronounced{i+1}" for i in range(29)]
                writer.writerow(header)

    def add_word(self, original_word, mispronounced_variations):
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [original_word] + mispronounced_variations
            writer.writerow(row)

    def display_dataset(self):
        with open(self.csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)

class MispronouncedWordsTorchDataset(Dataset):
    def __init__(self, csv_file="mispronounced.csv"):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        original_word = self.data.iloc[idx, 0]
        mispronounced_variations = self.data.iloc[idx, 1:].tolist()

        sample = {"original_word": original_word, "mispronounced_variations": mispronounced_variations}
        return json.dumps(sample)  # Convert sample to JSON format

def main():
    dataset = MispronouncedWordsDataset()
    dataset.add_word("original_word1", ["mispronounced_word1_1", "mispronounced_word1_2", "mispronounced_word1_3"])
    print("Dataset:")
    dataset.display_dataset()
    
    torch_dataset = MispronouncedWordsTorchDataset()
    sample = torch_dataset[0]
    print("Sample:", sample)
    for i in range(len(torch_dataset)):
        sample = torch_dataset[i]
        print(f"Sample {i}: {sample}")

if __name__ == "__main__":
    main()
