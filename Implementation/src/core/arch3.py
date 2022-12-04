# using pszemraj/long-t5-tglobal-base-16384-book-summary

from google.colab import drive
drive.mount('/content/gdrive')

dataset_path = 'gdrive/My Drive/fyp/cnn_dailymail/'

# Installing the necessary libraries
# !pip install transformers
# !pip install sentencepiece
# !pip install datasets
# !pip install optuna
# !pip install torch
# !pip install rouge_metric

# Importing the necessary libraries
import os
import torch
import random
import numpy as np
import pandas as pd
import datasets
import optuna
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

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
MIN_EPOCHS = 8
MAX_EPOCHS = 15
PER_DEVICE_EVAL_BATCH = 8
PER_DEVICE_TRAIN_BATCH = 8
MIN_BATCH_SIZE = 4
MAX_BATCH_SIZE = 8
NUM_TRIALS = 1
SAVE_DIR = 'opt-test'
SAVE_MODEL_DIR = 'cnn_dailymail-models'
SAVE_TOKENIZER_DIR = 'cnn_dailymail-tokenizer'
MODEL_NAME = 'pszemraj/long-t5-tglobal-base-16384-book-summary'
MAX_LENGTH = 512

# Load the dataset
print_custom('Loading the dataset')
dataset = pd.read_csv(dataset_path + 'cleaned_cnn_dailymail.csv', encoding='latin-1')

# Selecting the first 500 rows just to see if the GPU issue doesnt recreate as the dataset is large
dataset = dataset[0:500]

# Dataset shape
dataset.shape

#  Perform a train test split of 80:20 ratio on the dataset
print_custom('Performing train-test split on the dataset')
train_dataset = dataset[:int(len(dataset)*0.8)]
test_dataset = dataset[int(len(dataset)*0.8):]

# Creating a dataset Dict for the train and test dataset into a single dictionary
print_custom("Creating a dataset dict for the train and test split data")
datasetDict = datasets.DatasetDict({ 'train': datasets.Dataset.from_pandas(train_dataset), 'test': datasets.Dataset.from_pandas(test_dataset) })

# Loading the Tokenizer and Model
print_custom('Loading the Tokenizer and Model')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Preprocessing the dataset
prefix = "summarize: "
print_custom('Creating the tokenization function')
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

# Creating the optuna objective function for pszemraj/long-t5-tglobal-base-16384-book-summary model for summarization 
print_custom('Creating the optuna objective function for pszemraj/long-t5-tglobal-base-16384-book-summary model for summarization')
def objective(trial: optuna.Trial):
    # Specify the model name and folder
    model_path = f'{SAVE_MODEL_DIR}/{MODEL_NAME}'

    # Specify the training arguments and hyperparameter tune every arguments which are possible to tune
    training_args = Seq2SeqTrainingArguments(
        output_dir=SAVE_DIR,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=trial.suggest_float("learning_rate", LR_MIN, LR_CEIL, log=True),
        weight_decay=trial.suggest_float("weight_decay", WD_MIN, WD_CEIL, log=True),
        num_train_epochs=trial.suggest_int("num_train_epochs", MIN_EPOCHS, MAX_EPOCHS),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 1.0),
        per_device_train_batch_size=trial.suggest_int("per_device_train_batch_size", MIN_BATCH_SIZE, MAX_BATCH_SIZE),
        per_device_eval_batch_size=trial.suggest_int("per_device_eval_batch_size", MIN_BATCH_SIZE, MAX_BATCH_SIZE),
        save_total_limit=1,
        load_best_model_at_end=True,
        greater_is_better=True,
        predict_with_generate=True,
        run_name=MODEL_NAME,
        report_to="none",
    )


    # Create the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()

    # Return the loss
    return metrics["eval_loss"]

# Create the study
print_custom('Creating the study')
study = optuna.create_study(direction="minimize")

# Clearing the cuda memory
import torch
torch.cuda.empty_cache()

# Optimize the objective function
print_custom('Optimizing the objective function')
study.optimize(objective, n_trials=NUM_TRIALS)

# Print the best parameters
print_custom('Printing the best parameters')
print(study.best_params)

# Using the best parameters to train the model
print_custom('Using the best parameters to train the model')
training_args = Seq2SeqTrainingArguments(
    output_dir=SAVE_DIR,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=study.best_params["learning_rate"],
    weight_decay=study.best_params["weight_decay"],
    per_device_train_batch_size=study.best_params["per_device_train_batch_size"],
    per_device_eval_batch_size=study.best_params["per_device_eval_batch_size"],
    num_train_epochs=study.best_params["num_train_epochs"],
    warmup_ratio=study.best_params["warmup_ratio"],
    save_total_limit=1,
    load_best_model_at_end=True,
    greater_is_better=True,
    predict_with_generate=True,
    run_name=MODEL_NAME,
    report_to="none",
)

# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluating the model
trainer.evaluate()

# Model Evaluation using ROUGE metrics
print_custom('Making use of rouge metric to evaluate the model')
from rouge_metric import PyRouge

print_custom('Evaluating the model using rouge metric')
rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=True, rouge_s=True, rouge_su=True)

print_custom('Using the sample format to evaluate the model')
hypotheses = []
references = []

# Looping through the test dataset
for i in range(len(tokenized_dataset["test"])):
    # Getting the input and target
    input = tokenized_dataset["test"][i]["input_ids"]
    target = tokenized_dataset["test"][i]["labels"]

    # Decoding the input and target
    input = tokenizer.decode(input, skip_special_tokens=True)
    target = tokenizer.decode(target, skip_special_tokens=True)

    # Appending the input and target to the lists
    hypotheses.append(input)
    references.append([target])

# Evaluating the model
print_custom('Evaluating the model')
scores = rouge.evaluate(hypotheses, references)

# print the results
print_custom('Printing the results')
print(scores)

# Using the rouge score values, calculate the value in percentage for ROUGE-1, ROUGE-2 and ROUGE-L
print_custom('Using the rouge score values, calculate the value in percentage for ROUGE-1, ROUGE-2 and ROUGE-L')
rouge_1 = scores['rouge-1']['r'] * 100
rouge_2 = scores['rouge-2']['r'] * 100
rouge_l = scores['rouge-l']['r'] * 100

# Print the rouge score values
print_custom('Printing the rouge score values')
print(f'ROUGE-1: {round(rouge_1, 1)}')
print(f'ROUGE-2: {round(rouge_2, 1)}')
print(f'ROUGE-L: {round(rouge_l, 1)}')

# Save the model in the models folder with the name of the model
print_custom('Saving the model in the models folder with the name of the model')
trainer.save_model(f'{SAVE_MODEL_DIR}/{MODEL_NAME}')

# Download the model 
print_custom('Downloading the model')
from google.colab import files
files.download(f'{SAVE_MODEL_DIR}/{MODEL_NAME}')

# Save the tokenizer in the models folder with the name of the model
print_custom('Saving the tokenizer in the models folder with the name of the model')
tokenizer.save_pretrained(f'{SAVE_TOKENIZER_DIR}/{MODEL_NAME}')

# Download the tokenizer
print_custom('Downloading the tokenizer')
from google.colab import files
files.download(f'{SAVE_TOKENIZER_DIR}/{MODEL_NAME}')

# Save the study
print_custom('Saving the study')
import joblib
joblib.dump(study, f'{SAVE_DIR}/study.pkl')

# Download the study
print_custom('Downloading the study')
from google.colab import files
files.download(f'{SAVE_DIR}/study.pkl')

# Loading the model and tokenizer to make predictions
print_custom('Loading the model and tokenizer to make predictions')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(f'{SAVE_MODEL_DIR}/{MODEL_NAME}')
loaded_tokenizer = AutoTokenizer.from_pretrained(f'{SAVE_TOKENIZER_DIR}/{MODEL_NAME}')

# Testing out the model with the sample text 
print_custom('Testing out the model with the sample text')
# sample text
text = "I am a creative Full-Stack Web Developer who has experience in technologies such as Data Science & ML and Cloud Computing. I am a highly coordinated, committed and diplomatic software engineer with a defined capacity to operate and execute any specific role on schedule.I am able to communicate with a vast variety of individuals easily, with outstanding organizational skills. I see that I will bring my skills and expertise into practice in a full-time role in the industry, which will directly support the activities of the businesses I am involved in.I have the potential to build original conceptions and insights and solve a great many problems, guided by my intuitive and optimistic approach to problem solving. In algorithms as in business scenarios, I am able to apply my problems solving skills.Furthermore, I can easily and effectively understand the intensifying principles and help others to develop with great self encouragement. Therefore, I guess I am able to handle a lot of teams."

# make to sure to resolve the expected all tensors to be on the same device to be resolved when using the model on cpu 
import torch
print_custom('Resolving the expected all tensors to be on the same device to be resolved when using the model on cpu')
device = torch.device("cpu")
loaded_model.to(device)

# Tokenize the text
print_custom('Tokenizing the text')
inputs = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

# Generate the summary
print_custom('Generating the summary')
summary_ids = loaded_model.generate(inputs["input_ids"].to(device), num_beams=4, max_length=150, early_stopping=True)

# Decode the summary
print_custom('Decoding the summary')
summary = loaded_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the summary
print_custom('Printing the summary')
print(summary)