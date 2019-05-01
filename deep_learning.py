import math
import time
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from pylmnn import LargeMarginNearestNeighbor as LMNN
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0],28, 28,1)
print(X_train.shape)#(60000,28, 28,1)
#convert data type to float32
#normalize  data values to the range [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#grayscale normalization
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

model = Sequential()

model.add(Conv2D(32, 5, 5, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32,5,5,activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),name='feature_layer'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile model
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)



# Fit model on training data
start = time.time()
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=86),
                              epochs = 7, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 86
                              , callbacks=[learning_rate_reduction])
#history=model.fit(X_train,Y_train,epochs=30,batch_size=86,validation_split=0.33)
end = time.time()
print ("Model took %0.2f seconds to train"%(end - start))
#Model took 551.39 seconds to train
'''
f = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('#f Iterations')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
f.savefig("foo32_64_new.pdf", bbox_inches='tight')

f = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('#f Iterations')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
f.savefig("Model_loss_32.pdf", bbox_inches='tight')

'''

model.summary()

#Extract the hidden state representations
from keras.models import Model#, model_from_json
new_model=Model(inputs=model.input,outputs=model.get_layer('feature_layer').output)


from keras.models import model_from_json
new_model.save_weights("model_digit_new.h5")
print("Saved model to disk")
model_digit_json = new_model.to_json()
with open("model_digit_new.json", "w") as json_file:
    json_file.write(model_digit_json)
json_file.close()

MODEL = open('model_digit_new.json', 'r')
loaded_model_json = MODEL.read()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_digit_new.h5")
print("Loaded model from disk")
