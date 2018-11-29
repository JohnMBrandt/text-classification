import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import CustomObjectScope
from keras.engine.topology import Layer
from keras import initializers
from main import split_train
from model import Attention

base_dir = os.getcwd()
dir_x = os.path.join(base_dir, "ndc-extraction", "x")
dir_y = os.path.join(base_dir, "ndc-extraction", "y")
ls_x = os.listdir(dir_x)
ls_y = os.listdir(dir_y)

ls_x = [x for x in ls_x if ".txt" in x]
ls_y = [x for x in ls_y if ".txt" in x]

data_x = []
data_y = []

glove_dir = os.path.join(base_dir, "glove-embeddings")

def to_one_hot(labels, dim=2):
    results = np.zeros((len(labels), dim))
    for i, label in enumerate(labels):
        results[i][label - 1] = 1
    return results

def load_embeddings(glove_dir, word_index, max_words, embedding_dim):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return(embedding_matrix)

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
word_index = tokenizer.word_index

seq_x = []
for i in data_x:
    if len(i) < 100:
        i[len(i):100] = [''] * (100 - len(i))
    temp = (tokenizer.texts_to_sequences(i))
    temp = pad_sequences(temp, 30)
    seq_x.append(temp)
    
binary_y = []
for i in data_y:
    if len(i) < 100:
        i[len(i):100] = [0] * (100 - len(i))
    temp = [int(x) for x in i]
    #temp = to_one_hot(temp)
    binary_y.append(temp)

train_x, train_y, validation_x, validation_y, test_x, test_y = split_train(training_samples = 450,
                                                                            validation_samples = 1,
                                                                            test_samples = 50,
                                                                            shuffle = False,
                                                                            x = seq_x,
                                                                            y = binary_y)

embedding_weights = load_embeddings(glove_dir, word_index, 10000, 300)

class HierarchicalAttn():
    'Constructs a model with attention over words and Bidirectional GRU over sentences to encode sentences'
    def __init__(self, model, max_len, max_sentence, vocab_size, word_embedding, embedding_weights):
        self.model = None
        self.max_len = 30
        self.max_sentence = 100
        self.vocab_size = 10000
        self.embedding_weights = embedding_weights

    def weight_samples(self, train_y):
        samp_wt = np.zeros((len(train_y), self.max_sentence))
        for x, labs in enumerate(train_y):
            indiv_wt = np.zeros(self.max_sentence)
            for i, val in enumerate(train_y[x]):
                if val == 0:
                    indiv_wt[i] = 1
                else:
                    indiv_wt[i] = 5.6
            samp_wt[x] = indiv_wt
        return(samp_wt)
        
    def build_model(self, n_classes = 1, embedding_dim = 300):
        #embedding_weights = np.random.normal(0, 1, (10000, embedding_dim))
        l2_reg = regularizers.l2(1e-8)
        sentence_in = Input(shape = (self.max_len,), dtype = "int32")
        embedded_word_seq = Embedding(10000, 300, input_length = self.max_len, trainable = False, weights = [self.embedding_weights])(sentence_in)
        word_encoder = Bidirectional(GRU(50, return_sequences = True, kernel_regularizer = l2_reg, dropout = 0.2, recurrent_dropout = 0.2))(embedded_word_seq)
        dense_transform_w = Dense(100, activation = "relu", name = "dense_transform_w", kernel_regularizer = l2_reg)(word_encoder)
        attn_weighted_sent = Model(sentence_in, Attention(name = 'word_attention', regularizer = l2_reg)(dense_transform_w))
        attn_weighted_sent.summary()
        
        texts_in = Input(shape=(self.max_sentence, self.max_len), dtype='int32')
        attention_weighted_sentences = TimeDistributed(attn_weighted_sent)(texts_in)
        sentence_encoder = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=l2_reg))(attention_weighted_sentences)
        dense_transform_s = TimeDistributed(Dense(100, activation='relu', name='dense_transform_s',kernel_regularizer=l2_reg))(sentence_encoder) 
        #attention_weighted_text = TimeDistributed(Attention(name='sentence_attention', regularizer=l2_reg)(dense_transform_s))
        #flatten = TimeDistributed(Flatten())(sentence_encoder)
        prediction = TimeDistributed(Dense(1, activation = "sigmoid"))(dense_transform_s)
        #flatten = Flatten()(prediction)
        #pred_2 = TimeDistributed(Dense(1, activation = "sigmoid"))(flatten)
        model = Model(texts_in, prediction)
        model.summary()
        model.compile(optimizer = Adam(lr = 0.001), loss = "binary_crossentropy", metrics = ["acc"], sample_weight_mode = "temporal")
        return(model)
    
    def encode_texts(self, texts):
        'Reformat X data to be 3 dimensional array (docs, sentences, words)'
        encoded_texts = np.zeros((len(texts), self.max_sentence, self.max_len))
        for i, text in enumerate(texts):
            encoded_text = np.array(text)[:self.max_sentence]
            encoded_texts[i][-len(encoded_text):] = encoded_text
        return(encoded_texts)

    def encode_y(self, y):
        'Reformat Y data to be 3 dimensional array (docs, sentences, class)'
        encoded_ys = np.zeros((len(y), self.max_sentence))
        for i, text in enumerate(y):
            encoded_y = np.array(text)[:self.max_sentence]
            encoded_ys[i][-len(encoded_y):] = encoded_y
        encoded_ys = encoded_ys.reshape(len(y), 100, 1)
        return(encoded_ys)
    
    def train(self, train_x, train_y):
        'Encode x, y data and fit model'
        encoded_train_x = self.encode_texts(train_x)
        encoded_test_x = self.encode_texts(test_x)
        encoded_test_y = self.encode_y(test_y)
        encoded_train_y = self.encode_y(train_y)
        sample_weights = self.weight_samples(train_y)
        print(train_y[1])
        
        self.model = self.build_model()
        self.model.fit(encoded_train_x, encoded_train_y, epochs = 50, validation_split = 0.1, batch_size = 10,
                        sample_weight = sample_weights)
        preds = self.model.predict(encoded_test_x)
        print(preds)
        #test_y = [int(x) for x in test_y]
        #test_classes = probabilities.argmax(axis = -1) + 1
        #matrix = confusion_matrix(test_classes, test_y)
        #sess = tf.Session()
        #with sess.as_default():
        #   print(sess.run(matrix))


h = HierarchicalAttn(None, 30, 500, 10000, 300, embedding_weights)

h.train(train_x, train_y)