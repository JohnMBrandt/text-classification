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
    def __init(self):
        self.model = None
        self.max_len = 15
        self.max_sentence = 500
        self.vocab_size = 10000
        self.word_embedding = None
        
    def _build_model(self, n_classes = 1, embedding_dim = 300):
        l2_reg = regularizers.l2(1e-8)
        sentence_in = Input(shape = (15,), dtype = "int32")
        embedded_word_seq = Embedding(10000, 300, input_length = 15, trainable = True)(sentence_in)
        word_encoder = Bidirectional(GRU(50, return_sequences = True, kernel_regularizer = l2_reg))(embedded_word_seq)
        dense_transform_w = Dense(100, activation = "relu", name = "dense_transform_w", kernel_regularizer = l2_reg)(word_encoder)
        attn_weighted_sent = Model(sentence_in, Attention(name = 'word_attention', regularizer = l2_reg)(dense_transform_w))
        
        texts_in = Input(shape=(500, 15), dtype='int32')
        attention_weighted_sentences = TimeDistributed(attn_weighted_sent)(texts_in)
        sentence_encoder = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=l2_reg))(attention_weighted_sentences)
        dense_transform_s = Dense(100, activation='relu', name='dense_transform_s',kernel_regularizer=l2_reg)(sentence_encoder) 
        attention_weighted_text = Attention(name='sentence_attention', regularizer=l2_reg)(dense_transform_s)
        prediction = Dense(1, activation='sigmoid')(attention_weighted_text)
        self.model = Model(texts_in, prediction)
