import numpy
numpy.random.seed(100)

from keras.utils import to_categorical
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools

import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Reshape, LSTM, GRU
from keras.regularizers import l1

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from Plot_CM import plot_confusion_matrix
import pickle

print('Loading Data..')
X_train = numpy.load('Train_Data_S.npy')
X_test = numpy.load('Test_Data_S.npy')
X_valid = numpy.load('Valid_Data_S.npy')
y_train = numpy.load('Train_Label_S.npy')
y_test = numpy.load('Test_Label_S.npy')
y_valid = numpy.load('Valid_Label_S.npy')

print('Reshaping the Data that fit the model..')
X_train = numpy.squeeze(numpy.stack((X_train,) * 3, -1))
X_test = numpy.squeeze(numpy.stack((X_test,) * 3, -1))
X_valid = numpy.squeeze(numpy.stack((X_valid,) * 3, -1))

print('Train Data:{0}, Test Data:{1}, Valid Data:{2}'.format(X_train.shape,X_test.shape,X_valid.shape))
print('Train Label:{0}, Test Label:{1}, Valid Label:{2}'.format(y_train.shape,y_test.shape,y_valid.shape))

input_shape = X_train[0].shape
num_genres = 10

print('Input Shape:{0}'.format(input_shape))
print('Number of genres:{0}'.format(num_genres))

model = Sequential()
# Conv Block 1
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.25))

# Conv Block 2
model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.25))

# Conv Block 3
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.25))

# Conv Block 4
model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Dropout(0.25))

# Conv Block 5
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu',	padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.25))

# LSTM Block
model.add(Reshape((16,64)))
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(8,128)))
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(units=32, kernel_regularizer=l1(0.01), activation='relu'))
model.add(Dense(units=10, activation='softmax', kernel_regularizer=l1(0.01)))

# MLP
# model.add(Flatten())
# model.add(Dense(num_genres, activation='softmax'))

print('Summary..')
model.summary()

print('Compiling Model')
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print('Fitting Model')
hist = model.fit(X_train, y_train, batch_size=128, epochs=70, validation_data=(X_valid, y_valid), shuffle=False)

## Change the filename ##
print('Saving hist model into cnn_lstm_s.pickle')
with open('cnn_lstm_s.pickle','wb') as f:
	pickle.dump(hist.history, f)

model.save('Model_CNN_LSTM_S.h5')
####################################

score = model.evaluate(X_test, y_test, verbose=0)
print("Test_loss = {:.3f} and Test_acc = {:.3f}".format(score[0], score[1]))

# Plotting
plt.figure(figsize=(15,7))

plt.subplot(1,2,1)
plt.plot(hist.history['acc'], label='train')
plt.plot(hist.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print('Confusion Matrix Calculating..')
preds = numpy.argmax(model.predict(X_test), axis = 1)
y_orig = numpy.argmax(y_test, axis = 1)
cm = confusion_matrix(preds, y_orig)

genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()

plt.figure(figsize=(8,8))
plot_confusion_matrix(cm, keys, normalize=True)
plt.show()