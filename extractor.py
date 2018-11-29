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

base_dir = os.getcwd()

def split_train(training_samples, validation_samples, test_samples):
    'Split data into train, val, and test splits'
    x = seq_x
    y = binary_y
    #indices = np.arange(len(x))
    #np.random.shuffle(indices)
    #x = np.array(x)[indices]
    #y = np.array(y)[indices]
    train_x = x[ : training_samples]
    train_y = y[ : training_samples]
    validation_x = x[training_samples : training_samples + validation_samples]
    validation_y = y[training_samples : training_samples + validation_samples]
    test_x = x[training_samples + validation_samples : training_samples + validation_samples + test_samples]
    test_y = y[training_samples + validation_samples : training_samples + validation_samples + test_samples]
    return(train_x, train_y, validation_x, validation_y, test_x, test_y)

dir_x = os.path.join(base_dir, "ndc-extraction", "x")
dir_y = os.path.join(base_dir, "ndc-extraction", "y")
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
    if len(i) < 1054:
        i[len(i):1054] = [''] * (1054 - len(i))
    temp = (tokenizer.texts_to_sequences(i))
    temp = pad_sequences(temp, 15)
    seq_x.append(temp)


def to_one_hot(labels, dim=2):
    results = np.zeros((len(labels), dim))
    for i, label in enumerate(labels):
        results[i][label - 1] = 1
    return results
    
binary_y = []
for i in data_y:
    if len(i) < 1054:
        i[len(i):1054] = [0] * (1054 - len(i))
    temp = [int(x) for x in i]
    temp = to_one_hot(temp)
    binary_y.append(temp)

train_x, train_y, validation_x, validation_y, test_x, test_y = split_train(50, 20, 50)

class Attention(Layer):
    def __init__(self, regularizer=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.context = self.add_weight(name='context', 
                                       shape=(input_shape[-1], 1),
                                       initializer=initializers.RandomNormal(
                                            mean=0.0, stddev=0.05, seed=None),
                                       regularizer=self.regularizer,
                                       trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
        attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1)

        if mask is not None:
            # use only the inputs specified by the mask
            # import pdb; pdb.set_trace()
            attention = attention*K.cast(mask, 'float32')

        weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return (input_shape[0], input_shape[-1])

class HierarchicalAttn():
    def __init__(self, model, max_len, max_sentence, vocab_size, word_embedding):
        self.model = None
        self.max_len = 15
        self.max_sentence = 1054
        self.vocab_size = 10000
        self.word_embedding = None
        
    def build_model(self, n_classes = 1, embedding_dim = 300):
        embedding_weights = np.random.normal(0, 1, (10000, embedding_dim))
        l2_reg = regularizers.l2(1e-8)
        sentence_in = Input(shape = (self.max_len,), dtype = "int32")
        embedded_word_seq = Embedding(10000, 300, input_length = self.max_len, trainable = True, weights = [embedding_weights])(sentence_in)
        word_encoder = Bidirectional(GRU(50, return_sequences = True, kernel_regularizer = l2_reg))(embedded_word_seq)
        dense_transform_w = Dense(100, activation = "relu", name = "dense_transform_w", kernel_regularizer = l2_reg)(word_encoder)
        attn_weighted_sent = Model(sentence_in, Attention(name = 'word_attention', regularizer = l2_reg)(dense_transform_w))
        attn_weighted_sent.summary()
        
        texts_in = Input(shape=(self.max_sentence, self.max_len), dtype='int32')
        attention_weighted_sentences = TimeDistributed(attn_weighted_sent)(texts_in)
        sentence_encoder = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=l2_reg))(attention_weighted_sentences)
        #dense_transform_s = Dense(100, activation='relu', name='dense_transform_s',kernel_regularizer=l2_reg)(sentence_encoder) 
        #attention_weighted_text = Attention(name='sentence_attention', regularizer=l2_reg)(dense_transform_s)
        prediction = TimeDistributed(Dense(50))(attention_weighted_sentences)
        pred_2 = Dense(2, activation = "sigmoid")(prediction)
        model = Model(texts_in, pred_2)
        model.summary()
        model.compile(optimizer = Adam(lr = 0.001), loss = "binary_crossentropy", metrics = ["acc"])
        return(model)
    
    def encode_texts(self, texts):
        encoded_texts = np.zeros((len(texts), self.max_sentence, self.max_len))
        for i, text in enumerate(texts):
            encoded_text = np.array(text)[:self.max_sentence]
            encoded_texts[i][-len(encoded_text):] = encoded_text
        return(encoded_texts)

    def encode_y(self, y):
        encoded_ys = np.zeros((len(y), self.max_sentence, 2))
        for i, text in enumerate(y):
            encoded_y = np.array(text)[:self.max_sentence]
            encoded_ys[i][-len(encoded_y):] = encoded_y
        return(encoded_ys)
    
    def train(self, train_x, train_y):
        encoded_train_x = self.encode_texts(train_x)
        encoded_test_x = self.encode_texts(test_x)
        encoded_test_y = self.encode_y(test_y)
        encoded_train_y = self.encode_y(train_y)
        print(encoded_test_y[1])
        print(encoded_test_x.shape)
            #class_weights = {0: 1, 1: 20}
        self.model = self.build_model()
        self.model.fit(encoded_train_x, encoded_train_y, epochs = 2, validation_split = 0.1, batch_size = 10)
        preds = self.model.predict(encoded_test_x)
        print(preds)

h = HierarchicalAttn(None, 15, 500, 10000, 300)

h.train(train_x, train_y)