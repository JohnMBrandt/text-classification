from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from keras.models import Sequential, Model
from keras.metrics import top_k_categorical_accuracy
from tensorflow import confusion_matrix
from keras import regularizers
import tensorflow as tf
from keras.layers import Input, concatenate, Embedding, Dense, LSTM, TimeDistributed, Flatten, Activation, Bidirectional, Dropout, BatchNormalization, GRU
import math
import os
import numpy as np
import generator
from model import AttentionWithContext
from random import choices
from keras.callbacks import CSVLogger
from extractor import encode_sentences, load_data, HierarchicalAttn

def split_train(training_samples, validation_samples, test_samples, shuffle = True, x = None, y = None):
    'Split data into train, val, and test splits'
    if x == None:
        data = generator.load_obj("labels_dictionary")
        x = list(data.keys())
        y = list(data.values())
    if shuffle == True:
        # The "indices" file ensures that the training sentences come from the training documents
        # and the testing sentences come from the testing documents
        indices = generator.load_obj("classifier/indices")
        #indices = np.arange(len(x))
        #np.random.shuffle(indices)
        x = np.array(x)[indices]
        y = np.array(y)[indices]
    train_x = x[ : training_samples]
    train_y = y[ : training_samples]
    validation_x = x[training_samples : training_samples + validation_samples]
    validation_y = y[training_samples : training_samples + validation_samples]
    test_x = x[training_samples + validation_samples : training_samples + validation_samples + test_samples]
    test_y = y[training_samples + validation_samples : training_samples + validation_samples + test_samples]
    return(train_x, train_y, validation_x, validation_y, test_x, test_y)

def get_possibilities(i, train_y, rm):
	'Returns all training samples with i label'
	poss = []
	for x, ID in enumerate(train_y):
		e = [i for i in ID if i in rm]
		if len(e) == 0:
			if str(i + 1) in ID:
				poss.append(x)
	return(poss)

def add_augmented(train_x, train_y, thresh):
	'Augments training dataset by bootstrapping underrepresented classes'
	augmented_y = []
	augmented_x = []
	class_freq = []
	all_y_train = [item for sublist in train_y for item in set(sublist)]

	baseline = math.floor(len(all_y_train)/n_classes)
	print("Baseline {}".format(baseline))

	for i in range(1, n_classes + 1):
		class_freq.append(list(all_y_train).count(str(i)))
	class_prop = [round(x/baseline,2) for x in class_freq]
	oversampled = [x for x in set(all_y_train) if all_y_train.count(x) > 2*baseline]
	print("Not augmenting with labels from classes {} due to oversampling".format(oversampled))

	for i, val in enumerate(class_prop):
		if val < thresh:
			new_samples = math.floor(baseline * thresh - class_freq[i])
			print("Augmenting class {} with {} new samples".format(i + 1, new_samples))
			sample_choices = get_possibilities(i, train_y, oversampled)
			augmented_ids = np.random.choice(sample_choices, new_samples)
			augmented_y.append([train_y[i] for i in augmented_ids])
			augmented_x.append([train_x[i] for i in augmented_ids])

	augmented_y = [item for sublist in augmented_y for item in sublist]
	augmented_x = [item for sublist in augmented_x for item in sublist]
	return(augmented_x, augmented_y)
	
def load_embeddings(embed_dir, base_dir, max_words,
					word_index = None, embedding_dim = 300):
	'Calculate pre-trained gloVe embeddings for datat'
	print('### Loading {} dimensional GloVe embeddings for top {} words ###'.format(embedding_dim, max_words))
	if word_index is None:
		word_index = generator.load_obj("word_index")
	embeddings_index = {}
	f = open(os.path.join(base_dir, embed_dir, 'glove.6B.300d.txt'))
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

def make_model(embedding_matrix, n_classes):
	'Create model'
	l2_reg = regularizers.l2(1e-8)
	word_input = Input(shape = (max_len,), dtype = "int32")
	encoded_input = Input(shape = (100,), dtype = "float32")
	embedded_words = Embedding(10000, 300, input_length = max_len, trainable = False, weights = [embedding_matrix])(word_input)
	rnn = Bidirectional(GRU(50, return_sequences = True, dropout = 0.3, recurrent_dropout = 0.3, kernel_regularizer = l2_reg))(embedded_words)
	dense_w = TimeDistributed(Dense(100, kernel_regularizer = l2_reg))(rnn)
	attn = AttentionWithContext()(dense_w)

	merge_layer = concatenate([attn, encoded_input])
	dense_merged = Dense(100, activation = "relu", kernel_regularizer = l2_reg)(merge_layer)
	output = Dense(n_classes, activation = "sigmoid")(dense_merged)
	model = Model(inputs = [word_input, encoded_input], outputs = [output])
	model.summary()
	return(model)

def top_3_accuracy(y_true, y_pred):
	return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_1_accuracy(y_true, y_pred):
	return top_k_categorical_accuracy(y_true, y_pred, k=1)

def augment_and_weight_data(train_x, train_y, validation_x, validation_y, thresh, n_classes):
	print('### Augmenting under-represented classes with bootstrapped data ###')
	augmented_x, augmented_y = add_augmented(train_x, train_y, thresh)
	train_x = np.append(train_x, augmented_x)
	train_y = list(augmented_y) + list(train_y)

	class_freq = []
	all_y_train = [item for sublist in train_y for item in set(sublist)]
	for i in range(1, n_classes + 1):
		class_freq.append(all_y_train.count(str(i)))
	print("\n")
	for i in range(1, n_classes + 1):
		print('Training class ' + str(i) + ": {}".format(class_freq[i - 1]) + "      " + 
			'   Validation ' + str(i) + ": {}".format(list(validation_y).count(str(i))))
	print("\n")
	class_weights = {}
	for i in range(0, n_classes):
		count = all_y_train.count(str(i + 1))
		class_weights.update({i : round(1/count, 4)*1000})
	return(train_x, train_y, class_weights)

def main():
    print("\n### Pre-training sentence extractor ###")
    dir_x = os.path.join(base_dir, "ndc-extraction", "x")
    dir_y = os.path.join(base_dir, "ndc-extraction", "y")
    extr_x, extr_y, tokenizer = load_data(dir_x, dir_y)
    # Training on 335 samples will include up to "65-6.txt" and will include the first 5,035 sentences
    ex_train_x, ex_train_y, ex_validation_x, ex_validation_y, ex_test_x, ex_test_y = split_train(training_samples = 335,
                                                                            validation_samples = 0,
                                                                            test_samples = 115,
                                                                            shuffle = False,
                                                                            x = extr_x,
                                                                            y = extr_y)
    extraction_embeddings = load_embeddings(base_dir = base_dir, embed_dir = "glove-embeddings",
                                        word_index = tokenizer.word_index, max_words = 10000,)

    h = HierarchicalAttn(None, max_len = 30, max_sentence = 100, vocab_size = 10000, base_dir = base_dir,
                        embedding_weights = extraction_embeddings, train_x = ex_train_x, train_y = ex_train_y, test_x = ex_test_x, test_y = ex_test_y)
    h.train(ex_train_x, ex_train_y)

    print("### Training sentence classifier")

    print('\n### Loading parameter definitions ###')
    print('Batch size: {} \nEpochs: {} \n'.format(batch_size, epochs))
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_train(training_samples, validation_samples, test_samples)
    print('Training samples: {}\nValidation samples: {}\n Test samples: {}\n'.format(len(train_x), len(validation_x), len(test_x)))
    train_x, train_y, class_weights = augment_and_weight_data(train_x, train_y, validation_x, validation_y, thresh, n_classes = n_classes)

    print('\n### Creating generator and tokenizer ###')
    training_generator = generator.DataGenerator(train_x, train_x, train_y,
        batch_size = batch_size, n_classes = n_classes,
        max_words = max_words, max_len = max_len, base_dir = base_dir)
    validation_generator = generator.DataGenerator(validation_x, validation_x, validation_y,
        batch_size = batch_size, n_classes = n_classes,
        max_words = max_words, max_len = max_len, base_dir = base_dir)
    test_generator = generator.DataGenerator(test_x, test_x, test_y, batch_size = batch_size,
        n_classes = n_classes, max_words = max_words, max_len = max_len, base_dir = base_dir)

    embedding_matrix = load_embeddings(base_dir = base_dir, embed_dir = "glove-embeddings",
        max_words = max_words)

    classifier = make_model(embedding_matrix = embedding_matrix, n_classes = n_classes)
    print('\n### Compiling model with binary crossentropy loss and rmsprop optimizer ###')
    classifier.compile(loss = "binary_crossentropy",
        optimizer = "adam",
        metrics = [top_3_accuracy, top_1_accuracy])

    csv_logger = CSVLogger("log.csv", append = False, separator = ",")
    print("Class weights: {}".format(class_weights))
    classifier.fit_generator(generator = training_generator,
                   validation_data = validation_generator, epochs = epochs,
                   class_weight = class_weights, callbacks = [csv_logger])

    probabilities = classifier.predict_generator(generator = test_generator)
    generator.save_obj(probabilities, "pred_y")
    generator.save_obj(test_y, "true_y")

    print("\n### Saving model weights to simple_lstm.h5 ###")
    classifier.save_weights("simple_lstm.h5")

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = "main",
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dimension', default = 300, type = int)
	parser.add_argument('--epochs', default = 3, type = int)
	parser.add_argument('--batch_size', default = 100, type = int)
	parser.add_argument('--training_samples', default = 5032, type = int)
	parser.add_argument('--n_classes', default = 17, type = int)
	parser.add_argument('--validation_samples', default = 900, type = int)
	parser.add_argument('--test_samples', default = 800, type = int)
	parser.add_argument('--max_len', default = 50, type = int)
	parser.add_argument('--max_words', default = 10000, type = int)
	parser.add_argument('--thresh', default = 0.75, type = int)
	parser.add_argument('--base_dir', default = os.getcwd())
	args = parser.parse_args()

	dimension = args.dimension
	epochs = args.epochs
	batch_size = args.batch_size
	training_samples = args.training_samples
	test_samples = args.test_samples
	n_classes = args.n_classes
	validation_samples = args.validation_samples
	max_len = args.max_len
	thresh = args.thresh
	max_words = args.max_words
	base_dir = args.base_dir
	main()
