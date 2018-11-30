from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
import os
import numpy as np
import re
from collections import Counter
import pickle
from statistics import mean

# helper function to save files with pickle
def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# helper function to load files with pickle
def load_obj(name):
    with open("obj/" + name + ".pkl", "rb") as f:
        return pickle.load(f)

# data cleaning function
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
        string = re.sub(r",", "", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"!", "", string)
        string = re.sub(r'"', "", string)
        string = re.sub(r"\$", "", string)
        string = re.sub(r"\(", "", string)
        string = re.sub(r"\)", "", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

class DataGenerator(Sequence):
    def __init__(self, files, encoded, labels, batch_size = 32, 
        n_classes = 17, max_words = 10000, max_len = 50, base_dir = os.getcwd()):

        self.batch_size = batch_size
        self.labels = labels
        self.encoded = encoded
        self.files = files
        self.n_classes = n_classes
        self.max_words = max_words
        self.max_len = max_len
        self.shuffle = False
        self.base_dir = base_dir
        self.on_epoch_end()
        self.tokenizer = self.generate_vocab(os.path.join(self.base_dir, "ndc-data"))
        self.y_labs = self.load_labs("labels_dictionary")
        
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        files_temp = [self.files[k] for k in indexes]
        x, y = self.__data_generation(files_temp)
        return x, y

    def load_labs(self, name):
        with open("obj/" + name + ".pkl", "rb") as f:
            return pickle.load(f)

    def multihot(self, arr):
        trial_int = [int(i) - 1 for i in arr]
        y = np.zeros(17)
        for i in range(0, len(y) - 1):
            if i in trial_int:
                y[i] = 1
        return(y)

    def generate_vocab(self, dir):
        files = []
        for file in os.listdir(os.path.join(self.base_dir, dir)):
            file = open(os.path.join(self.base_dir, dir, file), encoding = "ISO-8859-1")
            cleaned = clean_str(file.read())
            files.append(cleaned)
        ##### NEW
        #files = set(files)
        l = Tokenizer(self.max_words)
        l.fit_on_texts(files)
        save_obj(l.word_index, "word_index")
        print('Created tokenizer with a {} vocab fit on {} documents padded to {} words'.format(self.max_words, len(files), self.max_len))
        return(l)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, files_temp):
        x = []
        x_interim = []
        y = np.empty(17)

        for ID in files_temp:
            file = open(os.path.join(self.base_dir, "ndc-data", str(ID) + ".txt"), encoding = "ISO-8859-1")
            cleaned = clean_str(file.read())
            y_temp = self.y_labs.get(ID)
            x.append(cleaned)
            file.close()
            y = np.vstack([y, [self.multihot(y_temp)]])
        y = y[1:]
        for ID in files_temp:
            file = np.loadtxt(os.path.join(self.base_dir, "ndc-encoded", str(ID) + ".txt"))
            x_interim.append(file)


        sequences = self.tokenizer.texts_to_sequences(x)
        x = pad_sequences(sequences, maxlen = self.max_len)
        x = [np.array(x), np.array(x_interim)]
        return x, y
