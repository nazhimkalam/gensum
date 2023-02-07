# Importing the necessary libraries
import torch
import numpy as np
import pandas as pd
import datasets
import optuna
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamWeightDecay, TFAutoModelForPreTraining
import tensorflow as tf
import os
from datasets import load_metric
import nltk
from huggingface_hub import notebook_login
from transformers.keras_callbacks import KerasMetricCallback

import transformers
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import pandas as pd
import datetime
import nltk


LR_MIN = 4e-5
LR_CEIL = 0.01
WD_MIN = 4e-5
WD_CEIL = 0.01
MIN_EPOCHS = 8
MAX_EPOCHS = 15
PER_DEVICE_EVAL_BATCH = 4
PER_DEVICE_TRAIN_BATCH = 4
MIN_BATCH_SIZE = 4
MAX_BATCH_SIZE = 6
NUM_TRIALS = 1
MAX_INPUT = 512
MAX_TARGET = 128
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'bart-base_model'
TOKENIZER_NAME = 'bart-base_tokenizer'

def print_custom(text):
    print('\n')
    print(text)
    print('-'*100)

def hyperparameter_serach(newData, userId, model, tokenizer, db):
    # note that we will be takening the model and tokenizer which is equal to the userId and train it on the new dataset
    # and then we will be saving the model and tokenizer in the same folder as the userId
    # and then we will be returning the model and tokenizer
    # nltk.download('punkt')
    metric = load_metric('rouge')
    
    # folder_path = 'model/' + userId
    # model_path =  folder_path + '/' + MODEL_NAME
    # tokenizer_path = folder_path + '/' + TOKENIZER_NAME
    
    # model.save_pretrained(model_path)
    # tokenizer.save_pretrained(tokenizer_path)
    # print('completed saving the model and tokenizer...')
    
    # return
    print(newData)
    
    def preprocess_data(data_to_process):
        inputs = [document for document in data_to_process['document']]

        #tokenize text
        model_inputs = tokenizer(inputs,  max_length=MAX_INPUT, padding='max_length', truncation=True)

        #tokenize labels
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(data_to_process['summary'], max_length=MAX_TARGET, padding='max_length', truncation=True)
            
        model_inputs['labels'] = targets['input_ids']
        return model_inputs
    
    print('created the preprocess data...')
    #  Perform a train test split of 80:20 ratio on the dataset (this case we need to assume that the new data is always a number more than 20 atleast)
    train_dataset = newData[:int(len(newData)*0.7)]
    test_dataset = newData[int(len(newData)*0.7):int(len(newData)*0.85)]
    validation_dataset = newData[int(len(newData)*0.85):]

    data = datasets.DatasetDict({ 'train': datasets.Dataset.from_pandas(train_dataset), 
                              'test': datasets.Dataset.from_pandas(test_dataset),
                              'validation': datasets.Dataset.from_pandas(train_dataset)})

    tokenize_data = data.map(preprocess_data, batched = True, remove_columns=['document', 'summary'])
    print("printing tokenized data: ", tokenize_data)
    
    # We are using batch_size to handle with the GPU limitation but if GPU size is not a limitation please use the recommend batch size from the hyperparameters
    batch_size = 1

    #data_collator to create batches. It preprocess data with the given tokenizer
    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
    
    print_custom('Performing hyperparameter training....')
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
            train_dataset=tokenize_data["train"],
            eval_dataset=tokenize_data["test"],
            tokenizer=tokenizer,
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        metrics = trainer.evaluate()

        torch.cuda.empty_cache()

        # Return the loss
        return metrics["eval_loss"]

    # Create the study
    print_custom('Creating the study')
    # study = optuna.create_study(direction="minimize")

    # Clearing the cuda memory
    import torch
    torch.cuda.empty_cache()

    # Optimize the objective function
    print_custom('Optimizing the objective function')
    # study.optimize(objective, n_trials=NUM_TRIALS)

    # Print the best parameters
    print_custom('Printing the best parameters')
    # print(study.best_params)
    
    # Hyperparameter results
    # learning_rate = study.best_params['learning_rate']
    # weight_decay = study.best_params['weight_decay']
    # num_train_epochs = study.best_params['num_train_epochs']
    # warmup_ratio = study.best_params['warmup_ratio']
    # per_device_train_batch_size = study.best_params['per_device_train_batch_size']
    # per_device_eval_batch_size = study.best_params['per_device_eval_batch_size']
    
    # hyperparameters = {
    #     learning_rate: learning_rate,
    #     weight_decay: weight_decay,
    #     num_train_epochs: num_train_epochs,
    #     warmup_ratio: warmup_ratio,
    #     per_device_train_batch_size: per_device_train_batch_size,
    #     per_device_eval_batch_size: per_device_eval_batch_size
    # }
    hyperparameters = {
        'learning_rate': 0.004114991060925223,
        'weight_decay': 6.461964495936167e-05,
        'num_train_epochs': 4,
        'warmup_ratio': 0.24666978823663988,
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4
    }
    
    model_retraining(hyperparameters, model, tokenizer, tokenize_data, data_collator, metric, db, userId)


def model_retraining(hyperparameters, model, tokenizer, tokenize_data, data_collator, metric, db, userId):
    os.environ["WANDB_DISABLED"] = "true"
    def compute_rouge(pred):
        predictions, labels = pred
        #decode the predictions
        decode_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #decode labels
        decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        #compute results
        res = metric.compute(predictions=decode_predictions, references=decode_labels, use_stemmer=True)
        #get %
        res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

        pred_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        res['gen_len'] = np.mean(pred_lens)

        return {k: round(v, 4) for k, v in res.items()}
    print('setting up the model re-training arguaments.....')
    args = transformers.Seq2SeqTrainingArguments(
        'model-retraining',
        evaluation_strategy='epoch',
        save_total_limit=2,
        # no_deprecation_warning=True,
        gradient_accumulation_steps=2,
        weight_decay=hyperparameters['weight_decay'],
        warmup_ratio=hyperparameters['warmup_ratio'],
        learning_rate=hyperparameters['learning_rate'],
        num_train_epochs=hyperparameters['num_train_epochs'],
        per_device_train_batch_size=1, # this is due to GPU limitation else hyperparameters.per_device_train_batch_size should be used 
        per_device_eval_batch_size= 1, # this is due to GPU limitation else hyperparameters.per_device_eval_batch_size should be used
        predict_with_generate=True,
        eval_accumulation_steps=1,
        fp16=False
    )
    
    trainer = transformers.Seq2SeqTrainer(
        model, 
        args,
        train_dataset=tokenize_data['train'],
        eval_dataset=tokenize_data['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_rouge
    )
     
    # Clearing the cuda memory
    import torch
    torch.cuda.empty_cache()
    
    trainer.train()
    
    # Clearing the cuda memory
    torch.cuda.empty_cache()

    # Evaluate the model
    metrics = trainer.evaluate()

    # Print the metrics
    print_custom('Printing the metrics')
    print(metrics)
    
    # saving the evaluations into the db
    saving_evaluations(metrics, db, userId)
    
    folder_path = 'model/' + userId
    model_path =  folder_path + '/' + MODEL_NAME
    tokenizer_path = folder_path + '/' + TOKENIZER_NAME
    
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
    

def saving_evaluations(metrics, db, userId):
    db.collection('domainUsers').document(userId).collection('model-retraining-evaluations').add({
        'created_at': datetime.datetime.now(),
        'eval_loss': metrics['eval_loss'],
        'eval_rouge1': metrics['eval_rouge1'],
        'eval_rouge2': metrics['eval_rouge2'],
        'eval_rougeL': metrics['eval_rougeL'],
        'eval_rougeLsum': metrics['eval_rougeLsum'],
        'eval_gen_len': metrics['eval_gen_len'],
        'eval_runtime': metrics['eval_runtime'],
        'eval_samples_per_second': metrics['eval_samples_per_second'],
        'eval_steps_per_second': metrics['eval_steps_per_second'],
        'epoch': metrics['epoch'],
    })
    print('Evaluations saved successfully into the db....')
    return True