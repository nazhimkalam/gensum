from google.colab import drive
drive.mount('/content/gdrive')

dataset_path = 'gdrive/My Drive/fyp/news_summary/'

# Installing the necessary libraries
# !pip install transformers
# !pip install sentencepiece
# !pip install datasets
# !pip install optuna
# !pip install rouge-metric
# !pip install torch

# Importing the necessary libraries
import os
import torch
import random
import numpy as np
import pandas as pd
import datasets
import optuna
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Create function for printing 
def print_custom(text):
    print('\n')
    print(text)
    print('-'*100)

# Specify our parameter and project variables
LR_MIN = 4e-5
LR_CEIL = 0.01
WD_MIN = 4e-5
WD_CEIL = 0.01
MIN_EPOCHS = 2
MAX_EPOCHS = 5
PER_DEVICE_EVAL_BATCH = 8
PER_DEVICE_TRAIN_BATCH = 8
NUM_TRIALS = 1
SAVE_DIR = 'opt-test'
SAVE_MODEL_DIR = 'models'
SAVE_TOKENIZER_DIR = 'tokenizer'
NAME_OF_MODEL = 'facebook/bart-large-cnn'
MAX_LENGTH = 512

# There are 2 datasets in the path dataset_path which are news_summary_more.csv and news_summary.csv, we will use news_summary_more.csv
# We will use the news_summary_more.csv dataset for training and validation

# Load the dataset
dataset = pd.read_csv(dataset_path + 'cleaned_news_summary.csv', encoding='latin-1')

#  Perform a train test split of 80:20 ratio on the dataset
train_dataset = dataset[:int(len(dataset)*0.8)]
test_dataset = dataset[int(len(dataset)*0.8):]

# Creating a DatasetDict for the train and test dataset into a single dictionary

datasetDict = datasets.DatasetDict({ 'train': datasets.Dataset.from_pandas(train_dataset), 'test': datasets.Dataset.from_pandas(test_dataset) })

# Loading the BartTokenizer and BartConfig
print_custom('Loading the BartTokenizer and BartConfig')
tokenizer = BartTokenizer.from_pretrained(NAME_OF_MODEL)
model = BartForConditionalGeneration.from_pretrained(NAME_OF_MODEL)

# Preprocessing the dataset
prefix = "summarize: "
print_custom('Tokenzing the dataset')
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenizing the dataset
tokenized_dataset = datasetDict.map(preprocess_function, batched=True)

# Creating a data Collector
print_custom('Creating a data Collector')
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Viewing the tokenized dataset structure
print_custom('Viewing the tokenized dataset structure')
print(tokenized_dataset)