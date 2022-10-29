# Hyperparameter tuning a transformer model with Optuna for text summarization

# Setting up imports
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
NAME_OF_MODEL = 'huggingoptunaface'
MAX_LENGTH = 512

# Work with news summary dataset (ctext column is the text to summarize and text column is the summary)
print_custom("Loading datasets....")
dataset_path = "./dataset/news_summary.csv"

# import dataset and split into train and test with the ratio of 80:20
dataset_train = load_dataset('csv', data_files=dataset_path, split='train[:80%]', features=datasets.Features({'ctext': datasets.Value('string'), 'text': datasets.Value('string')}),
encoding='latin-1')
dataset_test = load_dataset('csv', data_files=dataset_path, split='train[80%:]', features=datasets.Features({'ctext': datasets.Value('string'), 'text': datasets.Value('string')}), encoding='latin-1')

# Renaming columns
dataset_train = dataset_train.rename_column('text', 'summary')
dataset_train = dataset_train.rename_column('ctext', 'text')
dataset_test = dataset_test.rename_column('text', 'summary')
dataset_test = dataset_test.rename_column('ctext', 'text')

# Removing all the None values from the dataset
dataset_train = dataset_train.filter(lambda x: x['ctext'] is not None and x['text'] is not None)
dataset_test = dataset_test.filter(lambda x: x['ctext'] is not None and x['text'] is not None)

# Create a DatasetDict to hold our train and test datasets
dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})

# Loading the Electra small model
print_custom('Initializing Electra Small pretrained tokenizer')
model_name = "t5-small"  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  

prefix = "summarize: "

# Preprocess our text
print_custom('Preprocessing our text')
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Map our preprocessor to our dataset
print_custom('Mapping our preprocessor to our dataset')
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Creating a data collector
print_custom('Creating a data collector')
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors='tf')

# Using Optuna to set our objective function for text summarization where the text is the summary and the ctext is the text to summarize
print_custom('Setting our objective function for text summarization')
def objective(trial: optuna.Trial):
    # training_args = TrainingArguments(         
        # output_dir=SAVE_DIR, 
        # learning_rate=trial.suggest_loguniform('learning_rate', low=LR_MIN, high=LR_CEIL),         
        # weight_decay=trial.suggest_loguniform('weight_decay', WD_MIN, WD_CEIL),         
        # num_train_epochs=trial.suggest_int('num_train_epochs', low = MIN_EPOCHS,high = MAX_EPOCHS),         
        # per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,         
        # per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,         
        # disable_tqdm=True
    # )     

    # trainer = Trainer(
        # model=model,
        # args=training_args,
        # train_dataset=dataset['train'],
        # eval_dataset=dataset['test'])      
    
    # result = trainer.train()     
    # return result.training_loss

    training_args = Seq2SeqTrainingArguments(
        output_dir=SAVE_DIR, 
        weight_decay=trial.suggest_loguniform('weight_decay', WD_MIN, WD_CEIL),         
        learning_rate=trial.suggest_loguniform('learning_rate', low=LR_MIN, high=LR_CEIL),         
        num_train_epochs=trial.suggest_int('num_train_epochs', low = MIN_EPOCHS,high = MAX_EPOCHS),         
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,         
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,         
        evaluation_strategy='epoch',
        new_zeros=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,    
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        new_zeros=True,
    )

    result = trainer.train()
    return result.training_loss
    

# Create the Optuna study
#----------------------------------------------------------------------------------------------------
#                    CREATE OPTUNA STUDY
#----------------------------------------------------------------------------------------------------

print_custom('Triggering Optuna study')
# study = optuna.create_study(study_name='hp-search-electra', direction='minimize') 
# study.optimize(func=objective, n_trials=NUM_TRIALS)  
# trigger optuna study for abstractive text summarization
study = optuna.create_study(study_name='hp-search-t5', direction='minimize')
study.optimize(func=objective, n_trials=NUM_TRIALS)


# Get the best study hyperparameters
print_custom('Finding study best parameters....')
best_lr = float(study.best_params['learning_rate'])
best_weight_decay = float(study.best_params['weight_decay'])
best_epoch = int(study.best_params['num_train_epochs'])


print_custom('Extract best study params')
print(f'The best learning rate is: {best_lr}')
print(f'The best weight decay is: {best_weight_decay}')
print(f'The best epoch is : {best_epoch}')

print_custom('Create dictionary of the best hyperparameters')
best_hp_dict = {
    'best_learning_rate' : best_lr,
    'best_weight_decay': best_weight_decay,
    'best_epoch': best_epoch
}

# Create model based on best hyperparameters
#----------------------------------------------------------------------------------------------------
#                   TRAIN BASED ON OPTUNAS SELECTED HP
#----------------------------------------------------------------------------------------------------
 
print_custom('Training the model on the custom parameters')

training_args = TrainingArguments(         
    output_dir=SAVE_DIR, 
    learning_rate=best_lr,         
    weight_decay=best_weight_decay,         
    num_train_epochs=best_epoch,         
    per_device_train_batch_size=8,         
    per_device_eval_batch_size=8,         
    disable_tqdm=True)     

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test)      
    
result = trainer.train() 
trainer.evaluate()

# Saving our best model
print_custom('Saving the best Optuna tuned model')
if not path.exists('model'):
    os.mkdir('model')

model_path = "model/{}".format(NAME_OF_MODEL)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

