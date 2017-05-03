import os
import csv
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Convolution2D, Input, Flatten, Dense, MaxPooling2D, Activation, Dropout, Lambda
from sklearn.utils import shuffle
import cv2
import numpy as np
import sklearn
from keras.models import Model
import matplotlib.pyplot as plt

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

images=[]
measurements=[]
i=0

for line in samples:
    if i==0:
        i=1
        continue
    name = './IMG/'+ line[0].split('/')[-1]
    image = cv2.imread(name)
    angle = float(line[3])
    images.append(image)
    measurements.append(angle)

# trim image to only see section with road
X_train = np.array(images)
y_train = np.array(measurements)

#print(X_train.shape[1:])

# compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=32)
#validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col=3, 320, 160

#regression network, not using softmax for classification
model = Sequential()
#model.add(Flatten(input_shape=(160, 320, 3)))
#model.add(Dense(1))

#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
#model.add(Flatten(input_shape=(160, 320, 3)))
#convolution 1
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(90, 320, 3), output_shape=(90, 320, 3)))

model.add(Convolution2D(24, 3, 3, subsample=(2,2), activation="relu"))

#convolution 2
model.add(Convolution2D(36, 3, 3, subsample=(2,2), activation="relu"))

#convolution 3
model.add(Convolution2D(48, 3, 3, subsample=(2,2), activation="relu"))

#convolution 4
model.add(Convolution2D(64, 3, 3, activation="relu"))

#convolution 5
model.add(Convolution2D(64, 3, 3, activation="relu"))


model.add(Flatten(input_shape=(160, 320, 3)))

#fully connected layer 1, param of dense is the output size
model.add(Dense(50)) 
model.add(Activation('relu'))
model.add(Dropout(0.2))


#fcl 2
model.add(Dense(10)) 
model.add(Activation('relu'))
model.add(Dropout(0.2))


#fcl 3
model.add(Dense(1))

#mse is mean squared error
model.compile(loss='mse', optimizer='adam')
history_object=model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=25)

#model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=7)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('simple_model.h5')  


