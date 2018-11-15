from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, GRU, TimeDistributed, Flatten, Activation, Bidirectional, Dropout

import os
import numpy as np
import generator
import model

############################
### Variable definitions ###
############################

max_len = 500
max_words = 10000
training_samples = 3000
validation_samples = 1000
batch_size = 32

base_dir = '/Users/johnbrandt/Documents/python_projects/nlp_final'
ids = os.path.join(base_dir, 'parsed-reviews/collated/id_all.txt')
labels = os.path.join(base_dir, 'parsed-reviews/collated/label_3.txt')

def import_data(ids, labels):
	ids_list = []
	labs_list = []
	for line in open(ids):
	    ids_list.append(str.strip(line))
	for line in open(labels):
	    labs_list.append(str.strip(line))
	return({key:value for key, value in zip(ids_list, labs_list)})

def split_train(data, training_samples, validation_samples):
    keys = list(data.keys())
    values = list(data.values())
    print(values.count("0"), values.count("1"), values.count("2"))
    indices = np.arange(len(keys))
    np.random.shuffle(indices)
    keys = np.array(keys)[indices].tolist()
    values = np.array(values)[indices].tolist()
    print(len(keys))

    train_x = keys[ : training_samples]
    train_y = values[ : training_samples]
    
    validation_x = keys[training_samples : training_samples + validation_samples]
    validation_y = values[training_samples : training_samples + validation_samples]
    return(train_x, train_y, validation_x, validation_y)

############################################################################################
# make_model returns the following architecture:	     								   #
#	1. Embedding																	       #
#	2. GRU, returning H_i for each word Embedding 										   #
#	3. Dense relu on each word hidden state												   #
############################################################################################

def make_model():
    classifier = Sequential()
    classifier.add(Embedding(max_words, dimension, input_length = max_len))
    classifier.add(GRU(100, return_sequences = True, dropout = 0.2))
    classifier.add(TimeDistributed(Dense(50)))
    classifier.add(model.AttentionWithContext())
    classifier.add(Dense(3, activation = "softmax"))
    classifier.summary()
    return(classifier)

def main():
	n_classes = 3
	print('Batch size: {} \nEpochs: {} \n'.format(batch_size, epochs))
	data = import_data(ids, labels)

	train_x, train_y, validation_x, validation_y = split_train(data, training_samples, 1000)
	print('Training samples: {}\nValidation samples: {}\n'.format(len(train_x), len(validation_x)))

	for i in range(0, n_classes):
		print('Training class ' + str(i) + ": {}".format(train_y.count(str(i))) + "      " + 
			'   Validation ' + str(i) + ": {}".format(validation_y.count(str(i))))

	training_generator = generator.DataGenerator(train_x, train_y,
		batch_size = batch_size, n_classes = n_classes)
	validation_generator = generator.DataGenerator(validation_x, validation_y,
		batch_size = batch_size, n_classes = n_classes)

	classifier = make_model()
	classifier.compile(loss = "binary_crossentropy",
		optimizer = "rmsprop",
		metrics = ['categorical_accuracy'])

	print('\n\n\n')

	classifier.fit_generator(generator = training_generator,
                   validation_data = validation_generator, epochs = epochs)

	print("\n ### Saving model weights to simple_lstm.h5 ###")
	classifier.save_weights("simple_lstm.h5")


############################
###     Run if main      ###
############################

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = "main",
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dimension', default = 200, type = int)
	parser.add_argument('--epochs', default = 3, type = int)
	parser.add_argument('--batch_size', default = 32, type = int)
	parser.add_argument('--training_samples', default = 2000, type = int)
	args = parser.parse_args()

	dimension = args.dimension
	epochs = args.epochs
	batch_size = args.batch_size
	training_samples = args.training_samples
	main()

