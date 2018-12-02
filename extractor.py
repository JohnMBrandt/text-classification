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
import pickle
from generator import save_obj

base_dir = os.getcwd()
dir_x = os.path.join(base_dir, "ndc-extraction", "x")
dir_y = os.path.join(base_dir, "ndc-extraction", "y")
glove_dir = os.path.join(base_dir, "glove-embeddings")

def encode_sentences(weights):
    dir_y = os.listdir(os.path.join(base_dir, "ndc-extraction", "y"))
    dir_y = [x for x in dir_y if ".txt" in x]
    file = os.path.join(base_dir, "obj/merge/sdg_indexes.txt")
    classes = []
    html_i = []
    files = []
    f = open(file)
    for line in f:
        l = line.strip().split(" ")
        classes.append(l[0])
        html_i.append(int(l[1]) - (int(l[1])//100)*100)
        files.append(l[2])
    f.close()
    sent_encodings = np.array((len(sdg), 100))
    for i, val in enumerate(sdg):
        dim_1 = ls_y.index(files[i])
        dim_2 = html_i[i]
        current_encoding = sent_enc[dim_1, dim_2,]
        sent_encodings[i] = current_encoding
    return(sent_encodings)


def load_data(dir_x, dir_y):
    base_dir = os.getcwd()
    ls_x = [x for x in os.listdir(dir_x) if ".txt" in x]
    ls_y = [x for x in os.listdir(dir_y) if ".txt" in x]
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
        #temp = to_one_hot(temp)
        binary_y.append(temp)
    return(seq_x, binary_y, tokenizer)


def load_embeddings(glove_dir, word_index, max_words, embedding_dim):
    print('### Loading {} dimensional GloVe embeddings for top {} words ###'.format(embedding_dim, max_words))
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

class HierarchicalAttn():
    'Constructs a model with attention over words and Bidirectional GRU over sentences to encode sentences'
    def __init__(self, model, max_len, max_sentence, vocab_size, word_embedding, embedding_weights):
        self.model = None
        self.max_len = 30
        self.max_sentence = 100
        self.vocab_size = 10000
        self.embedding_weights = embedding_weights

    def weight_samples(self, train_y):
        'Weight binary classes based on their relative frequency'
        print("### Weighting samples ###")
        samp_wt = np.zeros((len(train_y), self.max_sentence))
        for x, labs in enumerate(train_y):
            indiv_wt = np.zeros(self.max_sentence)
            for i, val in enumerate(train_y[x]):
                if val == 0:
                    indiv_wt[i] = 1
                else:
                    indiv_wt[i] = 12
            samp_wt[x] = indiv_wt
        return(samp_wt)
        
    def build_model(self, n_classes = 1, embedding_dim = 300):
        'Build bi-level bi-directional GRU model with attention over word embeddings'
        l2_reg = regularizers.l2(1e-8)
        sentence_in = Input(shape = (self.max_len,), dtype = "int32")
        masking_layer = Masking(mask_value = 0)(sentence_in)
        embedded_word_seq = Embedding(10000, 300, input_length = self.max_len, trainable = False, weights = [self.embedding_weights])(masking_layer)
        word_encoder = Bidirectional(GRU(50, return_sequences = True, kernel_regularizer = l2_reg))(embedded_word_seq)
        dense_transform_w = Dense(100, activation = "relu", name = "dense_transform_w", kernel_regularizer = l2_reg)(word_encoder)
        attn_weighted_sent = Model(sentence_in, Attention(name = 'word_attention', regularizer = l2_reg)(dense_transform_w))
        attn_weighted_sent.summary()
        
        texts_in = Input(shape=(self.max_sentence, self.max_len), dtype='int32')
        attention_weighted_sentences = TimeDistributed(attn_weighted_sent)(texts_in)
        sentence_encoder = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=l2_reg), name = "sentence_encoder")(attention_weighted_sentences)
        dense_transform_s = TimeDistributed(Dense(100, activation='relu', name='dense_transform_s',kernel_regularizer=l2_reg))(sentence_encoder) 
        prediction = TimeDistributed(Dense(1, activation = "sigmoid"))(dense_transform_s)
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
        encoded_all = self.encode_texts(data_x)
        sample_weights = self.weight_samples(train_y)
        csv_logger = CSVLogger("log_encoder.csv", append = False, separator = ",")
        self.model = self.build_model()
        self.model.fit(encoded_train_x, encoded_train_y, epochs = 5, validation_split = 0.25, batch_size = 10,
                        sample_weight = sample_weights,  callbacks = [csv_logger])
        preds = self.model.predict(encoded_test_x)
        return(model)

        # Extract second LSTM output for the sentence representation to be transfered to sentence classifier
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer("sentence_encoder").output)
        intermediate_output = intermediate_layer_model.predict(encoded_all)
        encoded_sentences = encode_sentences(intermediate_output)
        print(encoded_sentences.shape)
        save_obj(encoded_sentences, "sentence_encoding")
        save_obj(preds, "sentence_extraction")

if __name__ == "__main__":
    data_x, data_y, tokenizer = load_data(dir_x, dir_y)
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_train(training_samples = 430,
                                                                            validation_samples = 1,
                                                                            test_samples = 10,
                                                                            shuffle = False,
                                                                            x = data_x,
                                                                            y = data_y)

    embedding_weights = load_embeddings(glove_dir, tokenizer.word_index, 10000, 300)
    h = HierarchicalAttn(None, 30, 500, 10000, 300, embedding_weights)
    h.train(train_x, train_y)