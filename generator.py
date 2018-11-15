from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
import os
import numpy as np
import re
from collections import Counter

base_dir = '/Users/johnbrandt/Documents/python_projects/nlp_final'

def clean_str(string):
        string = re.sub(r"[^A-Za-z0-9(),.!?_\"\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\"", " \" ", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'m", " \'m", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\$", " $ ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

class DataGenerator(Sequence):
    def __init__(self, files, labels, batch_size = 32, 
        n_classes = 3, max_words = 10000, max_len = 500):

        self.batch_size = batch_size
        self.labels = labels
        self.files = files
        self.n_classes = n_classes
        self.max_words = max_words
        self.max_len = max_len
        self.shuffle = False
        self.on_epoch_end()
        self.tokenizer = self.generate_vocab(base_dir)
        
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        files_temp = [self.files[k] for k in indexes]
        x, y = self.__data_generation(files_temp)
        return x, y

    def generate_vocab(self, dir):
        files = []
        for i, ID in enumerate(os.listdir(os.path.join(dir, "whole-reviews/collated"))):
            file = open(os.path.join(dir, "whole-reviews/collated/", ID), encoding = "ISO-8859-1")
            cleaned = clean_str(file.read())
            files.append(cleaned)
        l = Tokenizer(self.max_words)
        l.fit_on_texts(files)
        print('Created tokenizer with a {} vocab fit on {} documents'.format(self.max_words, len(files)))
        return(l)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, files_temp):
        x = []
        y = np.empty((self.batch_size), dtype = int)

        for i, ID in enumerate(files_temp):
            file = open(os.path.join(base_dir, "whole-reviews/collated/", ID + ".txt"), encoding = "ISO-8859-1")
            cleaned = clean_str(file.read())
            x.append(cleaned)
            file.close()
            y[i] = self.labels[self.files.index(ID)]

        #tokenizer = Tokenizer(num_words = self.max_words)
        #zxtokenizer.fit_on_texts(x)
        sequences = self.tokenizer.texts_to_sequences(x)
        x = pad_sequences(sequences, maxlen = self.max_len)
        return x, to_categorical(y, num_classes = self.n_classes)
