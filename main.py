from keras.models import Sequential
from keras.layers import Embedding, Dense, GRU, TimeDistributed, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
import os
import numpy as np
import generator

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
    indices = np.arange(len(data.keys()))
    np.random.shuffle(indices)

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
	model = Sequential()
	model.add(Embedding(max_words, dimension, input_length = max_len))
	model.add(GRU(100, return_sequences = True, dropout = 0.2))
	model.add(TimeDistributed(Dense(3, activation = 'relu')))
	model.add(Flatten())
	model.add(Dense(3, activation = "softmax"))
	model.summary()
	return(model)

def main():
	model = make_model()
	data = import_data(ids, labels)

	train_x, train_y, validation_x, validation_y = split_train(data, 2000, 1000)

	training_generator = generator.DataGenerator(train_x, train_y, batch_size = batch_size, n_classes = 3)
	validation_generator = generator.DataGenerator(validation_x, validation_y, batch_size = batch_size, n_classes = 3)

	model.compile(loss = "binary_crossentropy",
		optimizer = "rmsprop",
		metrics = ['acc'])

	print('\n\n\n')

	model.fit_generator(generator = training_generator,
                   validation_data = validation_generator, epochs = 3)

	print("Saving model weights to simple_lstm.h5")
	model.save_weights("simple_lstm.h5")


############################
###     Run if main      ###
############################

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = "main",
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dimension', default = 200, type = int)
	parser.add_argument('--epochs', default = 10, type = int)
	parser.add_argument('--batch_size', default = 32, type = int)
	args = parser.parse_args()

	dimension = args.dimension
	epochs = args.epochs
	batch_size = args.batch_size
	main()

