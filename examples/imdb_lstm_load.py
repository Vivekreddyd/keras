from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy
import pandas
import keras.callbacks

max_features = 20000
EMBEDDING_DIM=300
maxlen = 300  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
dev_file=pandas.read_csv('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/Venu/train.csv')
x_train = numpy.array(dev_file['text'])
#print (x_train)
y_train = numpy.array(dev_file['label'])

test_file=pandas.read_csv('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/Venu/test.csv')
x_test = numpy.array(test_file['text'])
#print (x_train)
y_test = numpy.array(test_file['label'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
word_index=tokenizer.word_index

print('Pad sequences (samples x time)')
#sequences_train=tokenizer.texts_to_sequences(x_train)
# x_train = sequence.pad_sequences(sequences_train,maxlen=maxlen)
sequences_test=tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(sequences_test,maxlen=maxlen)
# print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

encoder=LabelEncoder()
encoder.fit(y_test)
#encoded_ytrain=encoder.transform(y_test)
encoded_ytest=encoder.transform(y_test)

#convert integers to one hot encoded vectors
#onehot_ytrain=np_utils.to_categorical(encoded_ytrain)
onehot_ytest=np_utils.to_categorical(encoded_ytest)

#mapping words to indices of known embeddings
# embeddings_index = {}
# f = open('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/Word2Vec_GoogleNews-vectors-negative300.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = numpy.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# print('Found %s word vectors.' % len(embeddings_index))
#
# embedding_matrix = numpy.zeros((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#
#
# # to just use pretrained embeddings
# embedding_layer = Embedding(len(word_index) + 1,
#                                     EMBEDDING_DIM,
#                                     weights=[embedding_matrix],
#                                     input_length=maxlen,
#                                     trainable=False)

print('Build model...')
model = Sequential()
# model.add(embedding_layer)
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy','precision','recall','fmeasure'])

model.load_weights('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/valid/weights128_dp0.5_rdp=0.4_no_w2v_1500ep_fmeasure-01-0.7551.hdf5')

score = model.evaluate(numpy.array(x_test), encoded_ytest)

print('Test score:', score[0])
print('Test Accuracy:', score[1])
print('Test Precision:', score[2])
print('Test Recall:', score[3])
print('Test FMeasure:', score[4])

# model = load_model('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/valid/weights128_dp0.2_1500ep_fmeasure-05-0.7883.hdf5')