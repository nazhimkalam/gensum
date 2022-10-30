# importing gigaword dataset from tensorflow datasets 
import tensorflow_datasets as tfds
import tensorflow as tf

# loading the dataset
dataset, info = tfds.load('gigaword', with_info=True)

# print the dataset info
print(info)

# print the dataset
print(dataset)

# displaying the first 5 records of the dataset 
for i in dataset['train'].take(5):
    print(i)

# displaying the first 5 records of the dataset and only extract the 'document' column and "Summary" column and print it and convert the tensor to string 
for i in dataset['train'].take(5):
    print(i['document'])
    print(i['summary'])

# all the values are of type tensorflow.python.framework.ops.EagerTensor which I need to convert to string 
for i in dataset['train'].take(5):
    print(i['document'].numpy())
    print(i['summary'].numpy())

# creating a list to add the object of { 'document': 'summary': } 
data = []

# adding the object to the list
for i in dataset['train'].take(5):
    data.append({'document': i['document'].numpy(), 'summary': i['summary'].numpy()})

# print the list
print(data)

# covnerting the data into a csv file with the column names 'document' and 'summary' and downloading it from google colab
import pandas as pd
df = pd.DataFrame(data)
df.to_csv('gigaword.csv', index=False)

# downloading the csv file from google colab
from google.colab import files
files.download('gigaword.csv')

