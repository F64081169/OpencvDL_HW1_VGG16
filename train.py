import sys
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten

# load dataset
def load_dataset():
	(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
	Y_train = to_categorical(Y_train)
	Y_test = to_categorical(Y_test)
	return X_train, Y_train, X_test, Y_test

# scale pixels
def prepare_pixels(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def summarize_diag(history):
	# loss
	plt.subplot(211)
	plt.title('Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# accuracy
	plt.subplot(212)
	plt.title('Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_loss_and_accuracy_plot.png')
	plt.close()

def run_test():
	X_train, Y_train, X_test, Y_test = load_dataset()
	X_train, X_test = prepare_pixels(X_train, X_test)
	model = define_model()
	datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	it_train = datagen.flow(X_train, Y_train, batch_size=32)
	steps = int(X_train.shape[0] / 64)
	history = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_test, Y_test))
	model.save('model.h5')
	_, acc = model.evaluate(X_test, Y_test, verbose=0)
	print('> %.3f' % (acc * 100.0))
	summarize_diag(history)

# entry 
run_test()