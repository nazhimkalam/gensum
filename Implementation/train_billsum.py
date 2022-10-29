from pyexpat import features
import datasets 
import optuna 
from datasets import load_dataset 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os.path
from os import path
from datasets import load_dataset
# USING OPTUNA TO FIND THE BEST HYPERPARAMETERS FOR T-5 MODEL FOR SUMMARIZATION

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
NAME_OF_MODEL = 'huggingoptunaface'
MAX_LENGTH = 512

# Loading dataset
billsum = load_dataset("billsum", split="ca_test") 
billsum = billsum.train_test_split(test_size=0.2)
billsum = billsum.filter(lambda x: x['text'] is not None and x['summary'] is not None)
billsum["train"][0]

# drop title column
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

    # Specify the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=SAVE_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
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

