# Using Billsum dataset with optuna for text summarization

# Installing the necessary libraries
# !pip install transformers
# !pip install sentencepiece
# !pip install datasets
# !pip install optuna
# !pip install rouge-metric

from pyexpat import features
import datasets 
import optuna 
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os.path
from os import path

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
NAME_OF_MODEL = 'huggingoptunaface'
MAX_LENGTH = 512

# Loading dataset
# BillSum dataset will contain 3 columns which are text, summary and title
# I need to drop the 'title' column because it is not needed for the summarization task
billsum = load_dataset("billsum", split="ca_test") 
billsum = billsum.remove_columns(["title"])
billsum = billsum.train_test_split(test_size=0.2)
billsum = billsum.filter(lambda x: x['text'] is not None and x['summary'] is not None)
billsum["train"][0]

# create a dataset with the text and summary columns only
billsum = billsum.map(lambda x: {'text': x['text'], 'summary': x['summary']}, batched=True)
billsum["train"][0]  

# Dataset structure check
print_custom('Dataset structure check')
print(billsum)

# Loading t5-small model for summarization
print_custom('Initializing T5 Small pretrained tokenizer')
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Preprocessing the data
prefix = "summarize: "
print_custom('Tokenizing the dataset')
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    
# Tokenize the dataset
tokenized_billsum = billsum.map(preprocess_function, batched=True)

# Craeting a data collectior
print_custom('Creating a data collector')
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Viewing the tokenized dataset structure
print_custom('Tokenized dataset structure')
print(tokenized_billsum)

# Creating the optuna objective function for t5-small model for summarization 
print_custom('Creating the optuna objective function for t5-small model for summarization')
def objective(trial: optuna.Trial):
    # Specify the model name and folder
    model_name = "t5-small"
    model_folder = "model"
    model_path = f'{model_folder}/{model_name}'

    # Specify the training arguments and hyperparameter tune every arguments which are possible to tune
    training_args = Seq2SeqTrainingArguments(
        output_dir=SAVE_DIR,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=trial.suggest_float("learning_rate", LR_MIN, LR_CEIL, log=True),
        weight_decay=trial.suggest_float("weight_decay", WD_MIN, WD_CEIL, log=True),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        num_train_epochs=trial.suggest_int("num_train_epochs", MIN_EPOCHS, MAX_EPOCHS),
        save_total_limit=1,
        load_best_model_at_end=True,
        greater_is_better=True,
        predict_with_generate=True,
        run_name=NAME_OF_MODEL,
        report_to="none",
    )


    # Create the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
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
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    num_train_epochs=study.best_params["num_train_epochs"],
    save_total_limit=1,
    load_best_model_at_end=True,
    greater_is_better=True,
    predict_with_generate=True,
    run_name=NAME_OF_MODEL,
    report_to="none",
)

# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model using ROUGE metric and print the results 
print_custom('Evaluating the model using ROUGE metric and printing the results')
trainer.evaluate()

# Save the model in the models folder with the name of the model
print_custom('Saving the model in the models folder with the name of the model')
trainer.save_model(f'{SAVE_MODEL_DIR}/{NAME_OF_MODEL}')

# Save the tokenizer in the models folder with the name of the model
print_custom('Saving the tokenizer in the models folder with the name of the model')
tokenizer.save_pretrained(f'{SAVE_TOKENIZER_DIR}/{NAME_OF_MODEL}')

# Save the study
print_custom('Saving the study')
import joblib
joblib.dump(study, f'{SAVE_DIR}/study.pkl')

# Making use of rouge metric to evaluate the model  ==========================
print_custom('Making use of rouge metric to evaluate the model')
from rouge_metric import PyRouge

print_custom('Evaluating the model using rouge metric')
rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=True, rouge_s=True, rouge_su=True)

# sample hypothesis and reference
# hypotheses = [
#     'how are you\ni am fine',  # document 1: hypothesis
#     'it is fine today\nwe won the football game',  # document 2: hypothesis
# ]
# references = [[
#     'how do you do\nfine thanks',  # document 1: reference 1
#     'how old are you\ni am three',  # document 1: reference 2
# ], [
#     'it is sunny today\nlet us go for a walk',  # document 2: reference 1
#     'it is a terrible day\nwe lost the game',  # document 2: reference 2
# ]]

# Using the sample format to evaluate the model
print_custom('Using the sample format to evaluate the model')
hypotheses = []
references = []

# Looping through the test dataset
for i in range(len(tokenized_billsum["test"])):
    # Getting the input and target
    input = tokenized_billsum["test"][i]["input_ids"]
    target = tokenized_billsum["test"][i]["labels"]

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

# Save the results
print_custom('Saving the results')
import joblib
joblib.dump(scores, f'{SAVE_DIR}/scores.pkl')

# Loading the model and tokenizer to make predictions
print_custom('Loading the model and tokenizer to make predictions')
from transformers import T5ForConditionalGeneration, T5Tokenizer
loaded_model = T5ForConditionalGeneration.from_pretrained(f'{SAVE_MODEL_DIR}/{NAME_OF_MODEL}')
loaded_tokenizer = T5Tokenizer.from_pretrained(f'{SAVE_TOKENIZER_DIR}/{NAME_OF_MODEL}')

# Testing out the model with the sample text 
print_custom('Testing out the model with the sample text')
# sample text
text = "Summarization creates a shorter version of a document or an article that captures all the important information. Along with translation, it is another example of a task that can be formulated as a sequence-to-sequence task."

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


# https://colab.research.google.com/drive/1vvbkH6ZBrYI1hxxIsdtk5_C35dai6UaV?usp=sharing

# Hyperparameter list to be tuned 
# 1. Learning rate ✅
# 2. Weight Decay ✅
# 3. Number of epochs ✅
# 4. Batch size (This needs to be tested with later) ✅
# 5. Warmup ratio (This needs to be tested with aswell later) ✅


# https://huggingface.co/docs/transformers/tasks/summarization