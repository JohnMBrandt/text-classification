import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def split_train(training_samples, validation_samples, test_samples):
    'Split data into train, val, and test splits'
    x = seq_x
    y = binary_y
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = np.array(x)[indices]
    y = np.array(y)[indices]
    train_x = x[ : training_samples]
    train_y = y[ : training_samples]
    validation_x = x[training_samples : training_samples + validation_samples]
    validation_y = y[training_samples : training_samples + validation_samples]
    test_x = x[training_samples + validation_samples : training_samples + validation_samples + test_samples]
    test_y = y[training_samples + validation_samples : training_samples + validation_samples + test_samples]
    return(train_x, train_y, validation_x, validation_y, test_x, test_y)

dir_x = "/Users/johnbrandt/Documents/python_projects/nlp_final/ndc-extraction/x/"
dir_y = "/Users/johnbrandt/Documents/python_projects/nlp_final/ndc-extraction/y/"
ls_x = os.listdir(dir_x)
ls_y = os.listdir(dir_y)

ls_x = [x for x in ls_x if ".txt" in x]
ls_y = [x for x in ls_y if ".txt" in x]

data_x = []
data_y = []

for file in ls_x:
    f = open(os.path.join(dir_x, file))
    temp = []
    for line in f:
        temp.append(line.strip())
    f.close()
    data_x.append(temp)
    
for file in ls_y:
    f = open(os.path.join(dir_y, file))
    temp = []
    for line in f:
        temp.append(line.strip())
    f.close()
    data_y.append(temp)

tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(data_x)
seq_x = []
for i in data_x:
    temp = (tokenizer.texts_to_sequences(i))
    temp = pad_sequences(temp, 15)
    seq_x.append(temp)

binary_y = []
for i in data_y:
    temp = [int(x) for x in i]
    binary_y.append(temp)

train_x, train_y, validation_x, validation_y, test_x, test_y = split_train(115, 20, 20)
