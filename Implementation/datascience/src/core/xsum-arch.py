# Connecting to Gogole Colab
from google.colab import drive
drive.mount('/content/gdrive')

# Installing the required libraries
# !pip install transformers
# !pip install sentencepiece
# !pip install datasets
# !pip install optuna
# !pip install torch
# !pip install rouge_metric
# !pip install transformers datasets
# !pip install rouge-score nltk
# !pip install huggingface_hub

# Importing the necessary libraries
import torch
import numpy as np
import pandas as pd
import datasets
import optuna
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamWeightDecay, TFAutoModelForPreTraining
import tensorflow as tf
from datasets import load_metric
import nltk
from huggingface_hub import notebook_login
from transformers.keras_callbacks import KerasMetricCallback

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
MODEL_NAME = 't5-small'
MAX_LENGTH = 512

# Load the dataset
dataset_path = 'gdrive/My Drive/fyp/xsum/'
print_custom('Loading the dataset')
dataset = pd.read_csv(dataset_path + 'xsum.csv', encoding='latin-1')

# Selecting the first 100 rows just to see if the GPU issue doesnt recreate as the dataset is large
dataset = dataset[0:2000]

# Dataset shape
print(dataset.shape)

# Dataset head
print(dataset.columns) # ['document', 'summary']

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
    if 't5' in MODEL_NAME: 
        inputs = [prefix + doc for doc in examples["document"]]
    else:
        inputs = examples["document"]
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

# Creating the optuna objective function for sshleifer/distilbart-xsum-12-3 model for summarization 
print_custom('Creating the optuna objective function for sshleifer/distilbart-xsum-12-3 model for summarization')
def objective(trial: optuna.Trial):
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

model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)

# Print the best parameters
print_custom('Printing the best parameters')
print(study.best_params)

# Hyperparameter results
learning_rate = study.best_params['learning_rate']
weight_decay = study.best_params['weight_decay']
num_train_epochs = study.best_params['num_train_epochs']
warmup_ratio = study.best_params['warmup_ratio']
per_device_train_batch_size = study.best_params['per_device_train_batch_size']
per_device_eval_batch_size = study.best_params['per_device_eval_batch_size']

tokenized_dataset["train"]

# Preparing the datasets
train_dataset = model.prepare_tf_dataset(
    tokenized_dataset["train"],
    batch_size=per_device_train_batch_size,
    shuffle=True,
    collate_fn=data_collator,
)

validation_dataset = model.prepare_tf_dataset(
    tokenized_dataset["test"],
    batch_size=per_device_eval_batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

generation_dataset = model.prepare_tf_dataset(
    tokenized_dataset["test"],
    batch_size=8,
    shuffle=False,
    collate_fn=generation_data_collator
)

# Setting up the optimizer
optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)

# Loading Evaluation Metrics
metric = load_metric("rouge")

# a list to store the results of the evaluation
results = []

def metric_fn(eval_predictions):
    predictions, labels = eval_predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Rouge expects a newline after each sentence
    decoded_predictions = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_predictions
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]
    result = metric.compute(
        predictions=decoded_predictions, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    # Store the results
    results.append(result)

    return result

# Logging in to hugging face
# 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
notebook_login()

# Downloading the punkt tokenizer
nltk.download('punkt')

metric_callback = KerasMetricCallback(
    metric_fn, eval_dataset=generation_dataset, predict_with_generate=True, use_xla_generation=True
)

callbacks = [ metric_callback ]

# Training the model
model.fit(
    train_dataset, validation_data=validation_dataset, epochs=num_train_epochs, 
    callbacks=callbacks
)

# Testing the model with the test summary
document = 'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.\nRepair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.\nTrains on the west coast mainline face disruption due to damage at the Lamington Viaduct.\nMany businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.\nFirst Minister Nicola Sturgeon visited the area to inspect the damage.\nThe waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.\nJeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.\nHowever, she said more preventative work could have been carried out to ensure the retaining wall did not fail.\n"It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we\'re neglected or forgotten," she said.\n"That may not be true but it is perhaps my perspective over the last few days.\n"Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?"\nMeanwhile, a flood alert remains in place across the Borders because of the constant rain.\nPeebles was badly hit by problems, sparking calls to introduce more defences in the area.\nScottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.\nThe Labour Party\'s deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.\nHe said it was important to get the flood protection plan right but backed calls to speed up the process.\n"I was quite taken aback by the amount of damage that has been done," he said.\n"Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses."\nHe said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.\nHave you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.'

# for t5 models, we need to add the summarization prefix 
if 't5' in MODEL_NAME: 
    document = prefix + document

# Tokenize the document     
tokenized = tokenizer([document], return_tensors='np')

# Generate the summary
out = model.generate(**tokenized, max_length=MAX_LENGTH)

# Decode the summary and print it
with tokenizer.as_target_tokenizer():
    print(tokenizer.decode(out[0]))

# Save the model
drive_path_to_save_model = '/content/drive/MyDrive/Colab Notebooks/Models/' + MODEL_NAME + '_model'
model.save_pretrained(drive_path_to_save_model)

# Save the tokenizer
drive_path_to_save_tokenizer = '/content/drive/MyDrive/Colab Notebooks/Models/' + MODEL_NAME + '_tokenizer'
tokenizer.save_pretrained(drive_path_to_save_tokenizer)


# alternative way of creating train and test dataset using the data collator and th model than prepare_tf_dataset and also add the batch_size amd shuffle parameters to true and add the collate_fn parameter to the data collator. 

train_dataset = tf.data.Dataset.from_tensor_slices((tokenized_dataset["train"]["input_ids"], tokenized_dataset["train"]["attention_mask"], tokenized_dataset["train"]["labels"])).map(data_collator).batch(per_device_train_batch_size).shuffle(True)

# The above code is throwing 'Can't convert non-rectangular Python sequence to Tensor' error. 
# Hnadling the error by using the below code



# using tf.data.Dataset.from_tensor_slices to create the train dataset and test dataset
train_dataset = tf.data.Dataset.from_tensor_slices((tokenized_dataset["train"]["input_ids"], tokenized_dataset["train"]["attention_mask"], tokenized_dataset["train"]["labels"]))

# mountting google drive into kaggle notebook
from google.colab import drive
drive.mount('/content/drive')

# using the results array to store the results of rouge1, rouge2 and rougeL scores create a graph to show the results
import matplotlib.pyplot as plt
import numpy as np

rouge1 = []
rouge2 = []
rougeL = []

for i in range(len(results)):
    rouge1.append(results[i]['rouge1'])
    rouge2.append(results[i]['rouge2'])
    rougeL.append(results[i]['rougeL'])

x = np.arange(len(results))

plt.plot(x, rouge1, label='rouge1')
plt.plot(x, rouge2, label='rouge2')
plt.plot(x, rougeL, label='rougeL')

plt.xlabel('Epochs')
plt.ylabel('Scores')
plt.title('Rouge Scores')
plt.legend()
plt.show()

