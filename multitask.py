from model import AttentionWithContext, Attention
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from keras.models import Sequential, Model
from keras.metrics import top_k_categorical_accuracy
from tensorflow import confusion_matrix
from keras import regularizers
import tensorflow as tf
from keras.layers import *
import math
import os
import numpy as np
import pickle

base_dir = os.getcwd()

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# helper function to load files with pickle
def load_obj(name):
    with open("obj/" + name + ".pkl", "rb") as f:
        return pickle.load(f)
    
def multihot(arr):
        trial_int = [int(i) - 1 for i in arr]
        y = np.zeros(17)
        for i in range(0, len(y) - 1):
            if i in trial_int:
                y[i] = 1
        return(y)

def load_data(dir_x, dir_y):
    ls_x = [x for x in sorted(os.listdir(dir_x)) if ".txt" in x]
    ls_y = [x for x in sorted(os.listdir(dir_y)) if ".txt" in x]
    if ls_x == ls_y:
        print("X and Y data match")
    else:
        print("ERROR - X and Y data are mismatched")
    data_x = []
    data_y = []

    print("\n### Loading data ###")
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

    print("### Generating tokenizer ###")
    tokenizer = Tokenizer(num_words = 10000)
    tokenizer.fit_on_texts(data_x)
    word_index = tokenizer.word_index

    seq_x = []
    for i in data_x:
        if len(i) < 100:
            i[len(i):100] = [''] * (100 - len(i))
        temp = (tokenizer.texts_to_sequences(i))
        temp = pad_sequences(temp, 30, value = 0)
        seq_x.append(temp)
        
    binary_y = []
    for i in data_y:
        if len(i) < 100:
            i[len(i):100] = [0] * (100 - len(i))
        temp = [int(x) for x in i]
        binary_y.append(temp)
    return(seq_x, binary_y, tokenizer)



dir_x = os.path.join(base_dir, "ndc-extraction/x")
dir_y = os.path.join(base_dir, "ndc-extraction/y")
document_x, document_y, tokenizer = load_data(dir_x, dir_y)

ls_x = [x for x in sorted(os.listdir(dir_x)) if ".txt" in x]

file = os.path.join(base_dir, "sdg_indexes.txt")
class_data = []
html_i = []
files = []
f = open(file)
for line in f:
    l = line.strip().split(" ")
    class_data.append(int(l[0]))
    html_i.append(int(l[1]) - (int(l[1])//100)*100)
    files.append(l[2])
f.close()

document_x_expanded = []
document_y_expanded = []
for i, val in enumerate(ls_x):
    num = files.count(val)
    for l in range(0, num):
        document_x_expanded.append(document_x[l])
        document_y_expanded.append(document_y[l])

def encode_texts(texts):
        'Reformat X data to be 3 dimensional array (docs, sentences, words)'
        encoded_texts = np.zeros((len(texts), 100, 30))
        for i, text in enumerate(texts):
            encoded_text = np.array(text)[:100]
            encoded_texts[i][-len(encoded_text):] = encoded_text
        return(encoded_texts)
    
def encode_y(y):
        'Reformat Y data to be 3 dimensional array (docs, sentences, class)'
        encoded_ys = np.zeros((len(y), 100))
        for i, text in enumerate(y):
            encoded_y = np.array(text)[:100]
            encoded_ys[i][-len(encoded_y):] = encoded_y
        encoded_ys = encoded_ys.reshape(len(y), 100, 1)
        return(encoded_ys)

document_x = encode_texts(document_x_expanded)
document_y = encode_y(document_y_expanded)

class_y = load_obj("labels_dictionary")


sentence_x = []
sentence_y = np.empty(17)
sentence_id = []
for i in class_data:
    file_t = open("ndc-data/" + str(i-1) + ".txt", encoding = "ISO-8859-1")
    y_temp = class_y.get(i-1)
    sentence_id.append(i-1)
    cleaned = file_t.read()
    sentence_x.append(cleaned)
    sentence_y = np.vstack([sentence_y, [multihot(y_temp)]])
sentence_y = sentence_y[1:]

tokenizer =  Tokenizer(10000)
tokenizer.fit_on_texts(sentence_x)
sentence_x = tokenizer.texts_to_sequences(sentence_x)
sentence_x = pad_sequences(sentence_x, maxlen = 30)

def load_embeddings(max_words, word_index = None, embedding_dim = 300):
    'Calculate pre-trained gloVe embeddings for data'
    print('### Loading {} dimensional GloVe embeddings for top {} words ###'.format(embedding_dim, max_words))
    embeddings_index = {}
    f = open('glove-embeddings/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()

    word_index = word_index
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return(embedding_matrix)


embedding_matrix = load_embeddings(10000, tokenizer.word_index, 300)

'Build bi-level bi-directional GRU model with attention over word embeddings'
l2_reg = regularizers.l2(1e-8)

## Encode words for extraction
sentence_in = Input(shape = (30,), dtype = "int32")
embedded_word_seq = Embedding(10000, 300, input_length = 30, trainable = False, weights = [embedding_matrix])(sentence_in)
word_encoder = Bidirectional(GRU(50, return_sequences = True, kernel_regularizer = l2_reg))(embedded_word_seq)
dense_transform_w = Dense(100, activation = "relu", name = "dense_transform_w", kernel_regularizer = l2_reg)(word_encoder)
attn_weighted_sent = Model(sentence_in, Attention(name = 'word_attention', regularizer = l2_reg)(dense_transform_w))
attn_weighted_sent.summary()

def crop(id):
    def func(x):
        return x[:, id, :]
    return Lambda(func)


class_input = Input(shape = (30,), dtype = "int32", name = "CL_input")
class_ids = Input(shape = (1,), dtype = "int32", name = "CL_IDs")
# Sentence classifier

# Sentence extractor
texts_in = Input(shape=(100, 30), dtype='int32')
attention_weighted_sentences = TimeDistributed(attn_weighted_sent, name = "EX_sent_attn")(texts_in)
sentence_encoder = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=l2_reg, kernel_initializer='random_uniform',
                bias_initializer='zeros', name = "sentence_encoder"))(attention_weighted_sentences)
sentence_matcher = Lambda(lambda x: x[:,tf.squeeze(class_ids),:], output_shape=(100,))(sentence_encoder)
#sentence_matcher = crop(tf.squeeze(class_ids))(sentence_encoder).reshape((,100))
dense_transform_s = TimeDistributed(Dense(100, activation='relu', name='EX_sent_dense',kernel_regularizer=l2_reg))(sentence_encoder) 
output_extractor = TimeDistributed(Dense(1, activation = "sigmoid", name = "EX_out"))(dense_transform_s)


embedded_words = Embedding(10000, 300, input_length = 30, trainable = False, name = "CL_embed", weights = [embedding_matrix])(class_input)
rnn = Bidirectional(GRU(50, return_sequences = True, dropout = 0.3, recurrent_dropout = 0.3, name = "CL_RNN", kernel_regularizer = l2_reg))(embedded_words)
dense_w = TimeDistributed(Dense(100, kernel_regularizer = l2_reg, name = "CL_dense"))(rnn)
attn = AttentionWithContext(name = "CL_attn")(dense_w)
merge_layer = concatenate([attn, sentence_matcher], name = "CL_merging")
dense_merged = Dense(100, activation = "relu", kernel_regularizer = l2_reg, name = "CL_dense_2")(merge_layer)
output_classifier = Dense(17, activation = "sigmoid")(dense_merged)


model = Model(inputs = [class_input, texts_in, class_ids],  outputs = [output_classifier, output_extractor])
model.summary()

model.compile(optimizer = "adam", loss={'dense_1': 'binary_crossentropy'
                                        , 'time_distributed_2' : 'binary_crossentropy'},
             metrics = {'dense_1': [top_1_accuracy, top_3_accuracy], 'time_distributed_2' : ['acc']})

model.fit(x = [sentence_x, document_x, np.array(html_i)], y = [sentence_y, document_y],
          validation_split = 0.2, epochs = 10, batch_size = 1)