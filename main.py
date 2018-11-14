from keras.models import Sequential
from keras.layers import Embedding, Dense, GRU, TimeDistributed, Flatten
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import os
import numpy as np

############################
### Variable definitions ###
############################

max_len = 500
max_words = 10000
training_samples = 3000
validation_samples = 1000
batch_size = 32

############################
###      Directory       ###
############################

base_dir = '/Users/johnbrandt/Documents/python_projects/nlp_final'
whole_reviews = 'whole-reviews/'
authors = ['DennisSchwartz', 'JamesBerardinelli', 'ScottRenshaw', 'SteveRhodes']
whole_reviews_dir = os.path.join(base_dir, whole_reviews)
labels_dir = os.path.join(base_dir, "parsed-reviews/")

############################################################################################
# read_data reads in raw review data and classified labels (0,1,2) and does the following: #
# 	1. Generate list of reviews and classifications										   #
#   2. Tokenize and pad reviews                         								   #
#.  3. One-hot encode classifications													   #
############################################################################################

def read_data():
	print('\n####################### \n### Reading in data ###\n####################### \n\n')
	texts = []
	for author in authors: # Iterate through folders of author data
		author_dir = os.path.join(whole_reviews_dir, author, 'txt')
		for fname in os.listdir(author_dir):
		    if fname[-4:] == ".txt":
		        f = open(os.path.join(author_dir, fname), encoding = "ISO-8859-1") 
		        texts.append(f.read()) # Read in each review and append to text list
		        f.close()

	tokenizer = Tokenizer(num_words = max_words)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	data = pad_sequences(sequences, maxlen = max_len) # Pad reviews under 500 words

	labels = []
	for author in authors: # Iterate through folders of author data
		author_dir = os.path.join(labels_dir, author, 'label_3.txt')
		for line in open(author_dir):
		    labels.append(str.strip(line))
	print("Found {} files".format(len(texts)))
	labels = to_categorical(labels)
	return(data, labels)

############################################################################################
# split_train takes the data and labels to:												   #
#	1. Shuffle data to randomize													       #
#	2. Generate training and validation based on ArgumentParser 						   #
#	3. Return x_train, y_train, x_val, and y_val										   #
############################################################################################
        
def split_train(labels, data, data_size, training_samples, validation_samples):
	print('\n########################################### \n### Splitting train and validation data ###\n###########################################\n\n')
	indices = np.arange(data_size) 
	np.random.shuffle(indices) # Create randomized array from 1...data size
	data = data[indices] # Reindex data randomly to shuffle
	labels = np.array(labels)[indices] # Reindex labels randomly to shuffle

	x_train = data[:training_samples]
	y_train = labels[:training_samples]

	x_val = data[training_samples : training_samples + validation_samples]
	y_val = labels[training_samples : training_samples + validation_samples]
	print("There are {} training and {} validation samples of {} classes".format(y_train.shape[0], y_val.shape[0], y_train.shape[1]))
	return(x_train, y_train, x_val, y_val)

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
	(data, labels) = read_data()
	data_size = data.shape[0]
	(x_train, y_train, x_val, y_val) = split_train(labels, data, data_size, training_samples, validation_samples)
	model = make_model()

	model.compile(loss = "binary_crossentropy",
		optimizer = "rmsprop",
		metrics = ['acc'])

	print('\n\n\n')

	history = model.fit(x_train, y_train,
		batch_size = batch_size,
		epochs = epochs,
		validation_data =(x_val, y_val))

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
	args = parser.parse_args()


	dimension = args.dimension
	epochs = args.epochs

	main()

