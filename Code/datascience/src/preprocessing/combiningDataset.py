# Combine dataset
import pandas as pd
import os


dataset_paths = ["./datasetCombine/cnn_dailymail_cleaned_1.csv", 
                 "./datasetCombine/cnn_dailymail_cleaned_2.csv",
                 "./datasetCombine/cnn_dailymail_cleaned_3.csv",
                 "./datasetCombine/cnn_dailymail_cleaned_4.csv"]

dataset = pd.DataFrame()
for path in dataset_paths:
    dataset = dataset.append(pd.read_csv(path))

dataset.to_csv("./datasetCombine/cnn_dailymail_cleaned.csv", index=False)

