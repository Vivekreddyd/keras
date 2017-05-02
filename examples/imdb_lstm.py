'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy
import pandas
import keras.callbacks
from keras.utils import plot_model
import csv
max_features = 20000
MAX_NB_WORDS = 500
EMBEDDING_DIM=300
maxlen = 300  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

x_train = [];y_train=[];x_test=[];y_test=[];x_valid=[];y_valid=[]

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
dev_file=pandas.read_csv('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/Venu/train.csv')
'''dev_file = open('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/Venu/train.csv','rb')
dev_file_read=csv.reader(dev_file,delimiter=',',quotechar='"')
print (dev_file)
i=0
for row in dev_file_read:
    if(i==0):
        i+=1
    else:
        x_train.append(row[0])
        y_train.append(row[1])
'''
x_train = numpy.array(dev_file['text'])
#print (x_train)
y_train = numpy.array(dev_file['label'])
'''valid_file = open('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/Venu/test.csv','rb')
valid_file_read = csv.reader(dev_file, delimiter=',', quotechar='"')
i=0
for row in valid_file_read:
    if (i == 0):
        i += 1
    else:
        x_valid.append(row[0])
        y_valid.append(row[1])'''
test_file=pandas.read_csv('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/Venu/valid.csv')
x_test = numpy.array(test_file['text'])
#print (x_train)
y_test = numpy.array(test_file['label'])

'''test_file = open('/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/Venu/valid.csv','rb')
test_file_read = csv.reader(dev_file, delimiter=',', quotechar='"')
i=0
for row in test_file_read:
    if (i == 0):
        i += 1
    else:
        x_test.append(row[0])
        y_test.append(row[1])'''

#f_train  = open("/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.pos")
#f1_train = open("/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.neg")
#trainx_temp=f_train.readlines()
#trainx_temp1 = f1_train.readlines()

#trainy_temp = numpy.ones(len(trainx_temp))
#trainy_temp1 = numpy.zeros(len(trainx_temp1))


#x_train=f_train.readlines()
#test_temp = train_temp[4000:len(train_temp)]
#y_test = numpy.ones(len(x_test))

# y_train.append(y1_train)
# test=f1_train.readlines()
# x_train.append(test)
# x_test.append(x_train[8001:len(x_train)])
# y_test.append(numpy.zeros((len(x_test))-8000))
# y2_train = numpy.zeros(len(f1_train))
# y_train.append(y2_train)

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
word_index=tokenizer.word_index

# x_train = trainx_temp[0:4000] + trainx_temp1[0:4000]
# y_train = list(trainy_temp[0:4000]) + list(trainy_temp1[0:4000])
# length = len(y_train)
# x_test = trainx_temp[4001:len(trainx_temp)] + trainx_temp1[4001:len(trainx_temp1)]
# y_test = list(trainy_temp[4001:len(trainx_temp)]) + list(trainy_temp1[4001:len(trainx_temp1)])
#file.readlines()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
sequences_train=tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(sequences_train,maxlen=maxlen)
sequences_test=tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(sequences_test,maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# encode class value as integers
encoder=LabelEncoder()
encoder.fit(y_train)
encoded_ytrain=encoder.transform(y_train)
encoded_ytest=encoder.transform(y_test)

#convert integers to one hot encoded vectors
onehot_ytrain=np_utils.to_categorical(encoded_ytrain)
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

# print('Found %s word vectors.' % len(embeddings_index))

# embedding_matrix = numpy.zeros((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector


# to just use pretrained embeddings
# embedding_layer = Embedding(len(word_index) + 1,
#                                     EMBEDDING_DIM,
#                                     weights=[embedding_matrix],
#                                     input_length=maxlen,
#                                     trainable=False)

print('Build model...')
model = Sequential()
# model.add(embedding_layer)
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.4))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy','precision','recall','fmeasure'])
#x_train=list(x_train)
#y_train=list(y_train)
#x_test=list(x_test)
#y_test=list(y_test)
print('Train...')
checkpointer = ModelCheckpoint(filepath="/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/valid/weights128_dp0.5_rdp=0.4_no_w2v_1500ep_fmeasure-{epoch:02d}-{val_fmeasure:.4f}.hdf5", verbose=0, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=30)
tbCallBack = keras.callbacks.TensorBoard(log_dir='/local/vivek/keras/keras/datasets/cnn-text-classification-tf-master/data/Graphs', histogram_freq=0,
                            write_graph=True, write_images=True)
history = model.fit(numpy.array(x_train), encoded_ytrain,
          validation_data=(numpy.array(x_test), encoded_ytest),
          batch_size=batch_size,
          epochs=1500, callbacks=[checkpointer,earlystopping,tbCallBack])
score= model.evaluate(numpy.array(x_test), encoded_ytest,
                            batch_size=batch_size, verbose=0)

print('Test score:', score[0])
print('Test Accuracy:', score[1])
print('Test Precision:', score[2])
print('Test Recall:', score[3])
print('Test FMeasure:', score[4])
#print('Test accuracy:', score[2])

print (history.history['fmeasure'])
print (history.history['val_fmeasure'])
print (history.history['loss'])
print (history.history['val_loss'])
plt.plot(history.history['fmeasure'])
plt.plot(history.history['val_fmeasure'])
plt.title('model fmeasure')
plt.ylabel('fmeasure')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
