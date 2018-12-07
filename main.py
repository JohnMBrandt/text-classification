from model import AttentionWithContext, Attention, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.metrics import top_k_categorical_accuracy
from tensorflow import confusion_matrix
from keras import regularizers
import tensorflow as tf
from keras.layers import *
import math
import os
import numpy as np
import pickle

def top_3_accuracy(y_true, y_pred):
    'Top 3 accuracy metric for Keras'
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_1_accuracy(y_true, y_pred):
    'Top 1 accuracy metric for KEras'
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

def save_obj(obj, name):
    'Helper function using pickle to save and load objects'
    with open('results/' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    'Helper function using pickle to save and load objects'
    with open("results/" + name + ".pkl", "rb") as f:
        return pickle.load(f)
    
def multihot(arr):
    'Implementation of one-hot encoding for multi-labeled data'
    classes = [int(i) - 1 for i in arr]
    y = np.zeros(17)
    for i in range(0, len(y)):
        if i in classes:
            y[i] = 1
    return(y)

def load_data(dir_x, dir_y):
    'Loads and tokenizes document level data'
    ls_x = [x for x in sorted(os.listdir(dir_x)) if ".txt" in x]
    ls_y = [x for x in sorted(os.listdir(dir_y)) if ".txt" in x]
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
    doc_tokenizer = Tokenizer(num_words = max_words)
    doc_tokenizer.fit_on_texts(data_x)
    doc_word_index = doc_tokenizer.word_index
    seq_x = []
    binary_y = []

    for i in data_x:
        if len(i) < 100:
            i[len(i):100] = [''] * (100 - len(i))
        word_sequence = doc_tokenizer.texts_to_sequences(i)
        word_sequence = pad_sequences(word_sequence, sentence_len, value = 0)
        seq_x.append(word_sequence)

    for i in data_y:
        if len(i) < 100:
            i[len(i):100] = [0] * (100 - len(i))
        class_i = [int(x) for x in i]
        binary_y.append(class_i)

    return(seq_x, binary_y, doc_tokenizer)

def encode_texts(texts):
    'Reformat X data to be 3 dimensional array (docs, sentences, words)'
    encoded_texts = np.zeros((len(texts), 100, sentence_len))
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

def weight_data(train_y, n_classes):
    'Weights sentence classification loss function by class frequency'
    class_weights = {}
    for i in range(0, n_classes):
        count = sum(train_y[:,i])
        class_weights.update({i : round(7022/count,1)})
    return(class_weights)

def weight_samples(train_y):
    'Weights extractor loss function by class frequency'
    print("### Weighting samples ###")
    samp_wt = np.zeros((len(train_y), 100))
    for x, labs in enumerate(train_y):
        indiv_wt = np.zeros(100)
        for i, val in enumerate(train_y[x]):
            if val == 0:
                indiv_wt[i] = 1
            else:
                indiv_wt[i] = 4
        samp_wt[x] = indiv_wt
    return(samp_wt)

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

def build_model(sentence_len, max_words, doc_embedding, sent_embedding):
    'Constructs the multi-task training extractor/classifier model'
    l2_reg = regularizers.l2(1e-6)
    l1_l2reg = regularizers.l1_l2(1e-5)
    ## word encoder - extractor
    sentence_in = Input(shape = (sentence_len,), dtype = "int32")
    embedded_word_seq = Embedding(max_words, 300, input_length = sentence_len, trainable = False, 
                                weights = [doc_embedding])(sentence_in)
    #embedded_word_seq_learn = Embedding(10000, 300, input_length = 30, trainable = True)(sentence_in)
    #embedding_concat = concatenate([embedded_word_seq, embedded_word_seq_learn])
    word_encoder = Bidirectional(GRU(50, return_sequences = True, kernel_regularizer = l2_reg))(embedded_word_seq)
    dense_transform_w = Dense(100, activation = "relu", name = "dense_transform_w", kernel_regularizer = l2_reg)(word_encoder)
    attn_weighted_sent = Model(sentence_in, Attention(name = 'word_attention', regularizer = l2_reg)(dense_transform_w))
    attn_weighted_sent.summary()

    # Inputs - sentence encoder - extractor
    class_input = Input(shape = (sentence_len,), dtype = "int32", name = "CL_input")
    class_ids = Input(shape = (1,), dtype = "int32", name = "CL_IDs")
    texts_in = Input(shape=(100, sentence_len), dtype='int32')

    # sentence encoder - extractor
    attention_weighted_sentences = TimeDistributed(attn_weighted_sent, name = "EX_sent_attn")(texts_in)
    sentence_encoder = Bidirectional(GRU(50, return_sequences=True, name = "sentence_encoder"))(attention_weighted_sentences)
    sentence_matcher = Lambda(lambda x: x[:,tf.squeeze(class_ids),:], output_shape=(100,))(sentence_encoder)
    dense_transform_s = TimeDistributed(Dense(100, activation='relu', name='EX_sent_dense', 
                                            kernel_regularizer=l1_l2reg))(sentence_encoder) 
    dropout_extractor = Dropout(0.5)(dense_transform_s)
    output_extractor = TimeDistributed(Dense(1, activation = "sigmoid", name = "EX_out"))(dropout_extractor)

    # sentence classifier
    embedded_words = Embedding(max_words, 300, input_length = sentence_len, trainable = False, name = "CL_embed",
                            weights = [sent_embedding])(class_input)
    rnn = Bidirectional(GRU(50, return_sequences = True, name = "CL_RNN", kernel_regularizer = l2_reg))(embedded_words)
    dense_w = TimeDistributed(Dense(100, kernel_regularizer = l2_reg, name = "CL_dense"))(rnn)
    attn = AttentionWithContext(name = "CL_attn")(dense_w)
    merge_layer = concatenate([attn, sentence_matcher], name = "CL_merging")
    output_classifier = Dense(n_classes, activation = "sigmoid")(merge_layer)

    model = Model(inputs = [class_input, texts_in, class_ids],  outputs = [output_classifier, output_extractor])
    model.summary()
    model.compile(optimizer = Adam(lr = 0.0002),
                loss={'dense_1': 'binary_crossentropy', 'time_distributed_2' : 'binary_crossentropy'},
                metrics = {'dense_1': [top_1_accuracy, top_3_accuracy], 'time_distributed_2' : ['acc']})
    return(model)

def main():
    base_dir = os.getcwd()
    dir_x = os.path.join(base_dir, "data-summarization/x")
    dir_y = os.path.join(base_dir, "data-summarization/y")
    document_x, document_y, doc_tokenizer = load_data(dir_x, dir_y)
    ls_x = [x for x in sorted(os.listdir(dir_x)) if ".txt" in x]
    class_y = load_obj("labels_dictionary")

    file = os.path.join(base_dir, "data-summarization/merge_indices.txt")
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

    ## Broadcast document data
    document_x_expanded = []
    document_y_expanded = []
    for i, val in enumerate(ls_x):
        num = files.count(val)
        for l in range(0, num):
            document_x_expanded.append(document_x[i])
            document_y_expanded.append(document_y[i])

    document_x = encode_texts(document_x_expanded)
    document_y = encode_y(document_y_expanded)

    sentence_x = []
    sentence_y = np.empty(n_classes)
    sentence_id = []

    for i in class_data:
        file_t = open("data-classification/" + str(i-1) + ".txt", encoding = "ISO-8859-1")
        y_temp = class_y.get(i-1)
        sentence_id.append(i-1)
        cleaned = file_t.read()
        sentence_x.append(cleaned)
        sentence_y = np.vstack([sentence_y, [multihot(y_temp)]])
    sentence_y = sentence_y[1:]

    sent_tokenizer =  Tokenizer(max_words)
    sent_tokenizer.fit_on_texts(sentence_x)
    sentence_x = sent_tokenizer.texts_to_sequences(sentence_x)
    sentence_x = pad_sequences(sentence_x, maxlen = 30)

    ##### split train / val
    tr_sentence_x = sentence_x[0:training_samples]
    tr_doc_x = document_x[0:training_samples]
    tr_id = np.array(html_i[0:training_samples])
    tr_sentence_y = sentence_y[0:training_samples]
    tr_doc_y = document_y[0:training_samples]

    ts_sentence_x = sentence_x[training_samples:]
    ts_doc_x = document_x[training_samples:]
    ts_id = np.array(html_i[training_samples:])
    ts_sentence_y = sentence_y[training_samples:]
    ts_doc_y = document_y[training_samples:]

    doc_embedding_matrix = load_embeddings(max_words, doc_tokenizer.word_index, 300)
    sent_embedding_matrix = load_embeddings(max_words, sent_tokenizer.word_index, 300)
    class_weights = weight_data(ts_sentence_y, n_classes)
    sample_weights = weight_samples(ts_doc_y)

    model = build_model(sentence_len = sentence_len, max_words = max_words,
        sent_embedding = sent_embedding_matrix, doc_embedding = doc_embedding_matrix)

    model.fit(x = [tr_sentence_x, tr_doc_x, tr_id], y = [tr_sentence_y, tr_doc_y], shuffle = True, 
        sample_weight = {"ts_doc_y" : sample_weights}, class_weight = {"ts_sentence_y" : class_weights}, 
        epochs = epochs, batch_size = 1, validation_data = ([ts_sentence_x, ts_doc_x, ts_id], [ts_sentence_y, ts_doc_y]))

    preds = model.predict([ts_sentence_x, ts_doc_x, ts_id], batch_size = 1)
    save_obj(preds, "preds_multi_fixed")
    save_obj(tr_doc_y, "doc_y")
    save_obj(ts_sentence_y, "sentence_y")
    save_obj(ts_doc_y, "doc_y")


if __name__ == "__main__":
    ## Load in helper files
    import argparse
    parser = argparse.ArgumentParser(description = "main",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default = 2, type = int)
    parser.add_argument('--training_samples', default = 5000, type = int)
    parser.add_argument('--n_classes', default = 17, type = int)
    parser.add_argument('--sentence_len', default = 30, type = int)
    parser.add_argument('--max_words', default = 10000, type = int)
    parser.add_argument('--base_dir', default = os.getcwd())

    args = parser.parse_args()
    epochs = args.epochs
    training_samples = args.training_samples
    n_classes = args.n_classes
    sentence_len = args.sentence_len
    max_words = args.max_words

    main()
    