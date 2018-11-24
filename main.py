from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.metrics import top_k_categorical_accuracy
from tensorflow import confusion_matrix
import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM, TimeDistributed, Flatten, Activation, Bidirectional, Dropout, BatchNormalization
import math
import os
import numpy as np
import generator
import model
from random import choices

def split_train(training_samples, validation_samples, test_samples):
    data = generator.load_obj("labels_dictionary")
    x = list(data.keys())
    y = list(data.values())
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = np.array(x)[indices]
    y = np.array(y)[indices]
    train_x = x[ : training_samples]
    print(train_x)
    train_y = y[ : training_samples]
    validation_x = x[training_samples : training_samples + validation_samples]
    validation_y = y[training_samples : training_samples + validation_samples]
    test_x = x[training_samples + validation_samples : training_samples + validation_samples + test_samples]
    test_y = y[training_samples + validation_samples : training_samples + validation_samples + test_samples]
    return(train_x, train_y, validation_x, validation_y, test_x, test_y)

def get_possibilities(i, train_y):
	poss = []
	for x, ID in enumerate(train_y):
		if str(i + 1) in ID:
			poss.append(x)
	return(poss)

def add_augmented(train_x, train_y, thresh):
	augmented_y = []
	augmented_x = []
	class_freq = []
	all_y_train = [item for sublist in train_y for item in set(sublist)]

	baseline = math.floor(len(all_y_train)/n_classes)
	print("Baseline {}".format(baseline))

	for i in range(1, n_classes + 1):
		class_freq.append(list(all_y_train).count(str(i)))
	class_prop = [round(x/baseline,2) for x in class_freq]

	for i, val in enumerate(class_prop):
		if val < thresh:
			new_samples = math.floor(baseline * thresh - class_freq[i])
			print("Augmenting class {} with {} new samples".format(i + 1, new_samples))
			sample_choices = get_possibilities(i, train_y)
			augmented_ids = np.random.choice(sample_choices, new_samples)
			augmented_y.append([train_y[i] for i in augmented_ids])
			augmented_x.append([train_x[i] for i in augmented_ids])

	augmented_y = [item for sublist in augmented_y for item in sublist]
	augmented_x = [item for sublist in augmented_x for item in sublist]
	return(augmented_x, augmented_y)
	
def load_embeddings(embed_dir, name, base_dir, max_words,
					word_index = "word_index", embedding_dim = 300):
	print("\n### Calculating pre-trained gloVe embeddings ###")
	word_index = generator.load_obj(word_index)
	embeddings_index = {}
	f = open(os.path.join(base_dir, embed_dir, name))
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
	print("Saving glove embedding matrix\n")
	generator.save_obj(embedding_matrix, "embedding_matrix") 
	return(embedding_matrix)

def make_model(embedding_matrix, n_classes):
    classifier = Sequential()
    classifier.add(Embedding(max_words, dimension, input_length = max_len))
    classifier.add(Bidirectional(LSTM(50, return_sequences = True, dropout = 0.6, recurrent_dropout = 0.6)))
    classifier.add(BatchNormalization())
    classifier.add(TimeDistributed(Dense(50)))
    classifier.add(model.AttentionWithContext())
    classifier.add(Dense(n_classes, activation = "sigmoid"))

    classifier.layers[0].set_weights([embedding_matrix])
    classifier.layers[0].trainable = False
    classifier.summary()
    return(classifier)

def main():
	print('\n### Loading parameter definitions ###')
	print('Batch size: {} \nEpochs: {} \n'.format(batch_size, epochs))

	train_x, train_y, validation_x, validation_y, test_x, test_y = split_train(training_samples, validation_samples, test_samples)
	print('Training samples: {}\nValidation samples: {}\n'.format(len(train_x), len(validation_x)))
	
	class_freq = []
	all_y_train = [item for sublist in train_y for item in set(sublist)]
	for i in range(1, n_classes + 1):
		class_freq.append(all_y_train.count(str(i)))

	print("Sum of class freq {}".format(sum(class_freq)))
	for i in range(1, n_classes + 1):
		print('Training class ' + str(i) + ": {}".format(class_freq[i - 1]) + "      " + 
			'   Validation ' + str(i) + ": {}".format(list(validation_y).count(str(i))))
	print("\n")

	print('### Augmenting under-represented classes with bootstrapped data ###')
	augmented_x, augmented_y = add_augmented(train_x, train_y, thresh)
	train_x = np.append(train_x, augmented_x)
	train_y = list(augmented_y) + list(train_y)

	class_freq = []
	all_y_train = [item for sublist in train_y for item in set(sublist)]
	for i in range(1, n_classes + 1):
		class_freq.append(all_y_train.count(str(i)))

	print("Sum of class freq {}".format(sum(class_freq)))
	for i in range(1, n_classes + 1):
		print('Training class ' + str(i) + ": {}".format(class_freq[i - 1]) + "      " + 
			'   Validation ' + str(i) + ": {}".format(list(validation_y).count(str(i))))
	print("\n")

	print('\n### Creating generator and tokenizer ###')
	training_generator = generator.DataGenerator(train_x, train_y,
		batch_size = batch_size, n_classes = n_classes,
		 max_words = max_words, max_len = max_len, base_dir = base_dir)

	validation_generator = generator.DataGenerator(validation_x, validation_y,
		batch_size = batch_size, n_classes = n_classes,
		 max_words = max_words, max_len = max_len, base_dir = base_dir)

	test_generator = generator.DataGenerator(test_x, test_y, batch_size = batch_size,
		n_classes = n_classes, max_words = max_words, max_len = max_len, base_dir = base_dir)

	embedding_matrix = load_embeddings(base_dir = base_dir, embed_dir = "glove-embeddings",
		max_words = max_words, name = "glove.6B.300d.txt")

	classifier = make_model(embedding_matrix = embedding_matrix, n_classes = n_classes)
	print('\n### Compiling model with categorical crossentropy loss and rmsprop optimizer ###')
	#class_weights = []
	#for i in range(1, n_classes + 1):
	#	class_weights.append(list(train_y).count(str(i))/len(train_y))
	#print(class_weights)

	def top_3_accuracy(y_true, y_pred):
		return top_k_categorical_accuracy(y_true, y_pred, k=3)

	def top_1_accuracy(y_true, y_pred):
		return top_k_categorical_accuracy(y_true, y_pred, k=1)

	classifier.compile(loss = "binary_crossentropy",
		optimizer = "adam",
		metrics = [top_3_accuracy, top_1_accuracy])

	#class_weights = {}
	#for i in range(1, n_classes + 1):
	#	count = list(train_y).count(str(i))
	#	class_weights.update({i-1 : 1/count})
	classifier.fit_generator(generator = training_generator,
                   validation_data = validation_generator, epochs = epochs)

	probabilities = classifier.predict_generator(generator = test_generator)
	generator.save_obj(probabilities, "pred_y")
	generator.save_obj(test_y, "true_y")
	#test_y = [int(x) for x in test_y]
	#test_classes = probabilities.argmax(axis = -1) + 1
	#matrix = confusion_matrix(test_classes, test_y)
	#sess = tf.Session()
	#with sess.as_default():
	#	print(sess.run(matrix))
	print("\n### Saving model weights to simple_lstm.h5 ###")
	classifier.save_weights("simple_lstm.h5")

############################
###     Run if main      ###
############################

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = "main",
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dimension', default = 300, type = int)
	parser.add_argument('--epochs', default = 3, type = int)
	parser.add_argument('--batch_size', default = 100, type = int)
	parser.add_argument('--training_samples', default = 4000, type = int)
	parser.add_argument('--n_classes', default = 17, type = int)
	parser.add_argument('--validation_samples', default = 1500, type = int)
	parser.add_argument('--test_samples', default = 2000, type = int)
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
