import numpy
numpy.random.seed(12345)

from keras.utils import to_categorical
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools

import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, Reshape, LSTM, GRU, Bidirectional
from keras.regularizers import l1

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from Plot_CM import plot_confusion_matrix
import pickle


print('Loading Data..')
X_train = numpy.load('Train_Data.npy')
X_test = numpy.load('Test_Data.npy')
X_valid = numpy.load('Valid_Data.npy')
y_train = numpy.load('Train_Label.npy')
y_test = numpy.load('Test_Label.npy')
y_valid = numpy.load('Valid_Label.npy')

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

freezed_layers = 5
print('Building Model..')
input_tensor = Input(shape=input_shape)
W = VGG16(include_top=False, weights=None,input_tensor=input_tensor)
top = Sequential()
# top.add(Flatten(input_shape=W.output_shape[1:]))
top.add(Reshape([16,512]))
top.add(LSTM(128, return_sequences=True, recurrent_dropout=0.5, input_shape=(16,512)))
top.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5)))
top.add(LSTM(128, return_sequences=False))
top.add(Dense(32, activation='relu'))
top.add(Dense(num_genres, activation='softmax'))

model = Model(inputs=W.input, outputs=top(W.output))
for layer in model.layers[:freezed_layers]:
    layer.trainable = False

print('Summary..')
model.summary()

print('Compiling Model')
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print('Fitting Model')
hist = model.fit(X_train, y_train, batch_size=100, epochs=15, validation_data=(X_valid, y_valid))

print('Saving hist model into vgg16_lstm_hist_3.pickle')
with open('vgg16_lstm_hist_3.pickle','wb') as f:
	pickle.dump(hist.history, f)

model.save('Model_VGG16_LSTM_3.h5')

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