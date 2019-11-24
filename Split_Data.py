import numpy
from sklearn.model_selection import train_test_split
from keras import utils

X = numpy.load('X_Feat.npy')
y = numpy.load('y_Feat.npy')

y = utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.1, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,random_state=42,test_size=0.1, stratify=y_train)

print('Saving..')
numpy.save('Train_Data_S.npy',X_train)
numpy.save('Train_Label_S.npy',y_train)
numpy.save('Test_Data_S.npy',X_test)
numpy.save('Test_Label_S.npy',y_test)
numpy.save('Valid_Data_S.npy',X_valid)
numpy.save('Valid_Label_S.npy',y_valid)