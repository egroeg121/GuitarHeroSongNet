import idx
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#load data if not previously loaded and split into training and testing sets
try:
    training_data
except NameError:
    print("Reading data...")
    training_data=np.expand_dims(idx.read("train-images.idx3-ubyte"), axis=3)
    testing_data=training_data[0:15000,:,:,:]
    training_data=training_data[15000:-1,:,:,:]

try:
    training_labels
except NameError:
    print("Reading labels...")
    training_labels=to_categorical(idx.read("train-labels.idx1-ubyte"), num_classes=10)
    testing_labels=training_labels[0:15000,:]
    training_labels=training_labels[15000:-1,:]

#create model (VGG-like model)
inputshape=[]
inputshape[1:3]=training_data.shape[1:3]
inputshape.append(1)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=inputshape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

#train model
fit_history=model.fit(training_data, training_labels, batch_size=512, epochs = 150)
score = model.evaluate(testing_data, testing_labels, batch_size=512)

plt.plot(fit_history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(fit_history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()