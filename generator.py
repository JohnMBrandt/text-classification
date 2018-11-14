from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
import os
import numpy as np

base_dir = '/Users/johnbrandt/Documents/python_projects/nlp_final'


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
        
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        files_temp = [self.files[k] for k in indexes]
        x, y = self.__data_generation(files_temp)
        return x, y
    
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
            x.append(file.read())
            file.close()
            y[i] = self.labels[self.files.index(ID)]

        tokenizer = Tokenizer(num_words = self.max_words)
        tokenizer.fit_on_texts(x)
        sequences = tokenizer.texts_to_sequences(x)
        x = pad_sequences(sequences, maxlen = self.max_len)
        return x, to_categorical(y, num_classes = self.n_classes)
