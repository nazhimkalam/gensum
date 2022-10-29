from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os.path
from os import path
import numpy as np
from numpy import argmax

# Specify where to load the model and tokenizer from
MODEL_NAME = 'huggingoptunaface'
MODEL_FOLDER = 'model'
MODEL_PATH = f'{MODEL_FOLDER}/{MODEL_NAME}'
MAX_LENGTH = 512

# Load our model and tokenizer
loaded_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
loaded_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Our example text to pass to our fine tuned model
text = 'Acute generalized exanthematous'

def get_result(text, message=True):
    encoded_input = loaded_tokenizer(text, truncation=True, padding='max_length',
                                     max_length=MAX_LENGTH, return_tensors='pt')
    output = loaded_model(**encoded_input)
    result = output[0].detach().numpy()
    probs = torch.sigmoid(output[0]).detach().numpy()
    class_label = argmax(result)
    
    if message:
        print(f'The predicted class is label: {str(class_label)} with a probability of {probs[0][0]}')
    
    return result, class_label, probs


# Run your result through the function
result, class_label, probs = get_result(text)