import keras
import numpy as np
import handwritemodel as hwmodel
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import to_categorical


# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print( "Shape of train data:")
print(X_train.shape)
print( "Shape of test data:")
print(y_train.shape)

# Adjust incoming data
X_train_ready = np.expand_dims(X_train, 3)
y_train_ready = to_categorical(y_train)

X_test_ready = np.expand_dims(X_test, 3)
y_test_ready = to_categorical(y_test)

# Import model
model = hwmodel.getModel()

# compile
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Create logger object
logger = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True,histogram_freq=5)

# train
model.fit(
    X_train_ready,
    y_train_ready,
    epochs=10,
    batch_size=500,
    verbose=2,
    shuffle= True,
    callbacks=[logger],
    validation_data=[X_test_ready, y_test_ready]
)

# evaluate
#model.evaluate(  )