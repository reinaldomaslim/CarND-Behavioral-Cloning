import os
import csv
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Convolution2D, Input, Flatten, Dense, MaxPooling2D, Activation, Dropout
from keras.layers import Lambda
from sklearn.utils import shuffle
from keras.models import Model, load_model
import matplotlib.pyplot as plt

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=35):
    num_samples = len(samples)
    while 1: # Used as a reference pointer so code always loops back around
        shuffle(samples)
        for offset in range(1, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                correction=0.1

                #center image
                name = './data/IMG/'+ batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(-center_angle)


                #left camera
                name = './data/IMG/'+ batch_sample[1].split('/')[-1]
                left_image= cv2.imread(name)
                angle=center_angle+correction
                images.append(left_image)
                angles.append(angle)
                images.append(cv2.flip(left_image,1))
                angles.append(-angle)

                #right camera
                name ='./data/IMG/'+ batch_sample[2].split('/')[-1]
                right_image=cv2.imread(name)
                angle=center_angle-correction
                images.append(right_image)
                angles.append(angle)
                images.append(cv2.flip(right_image,1))
                angles.append(-angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            X_train, y_train=shuffle(X_train, y_train)
            yield X_train, y_train

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=35)
validation_generator = generator(validation_samples, batch_size=35)



model = load_model('model.h5')




#mse is mean squared error
model.compile(loss='mse', optimizer='adam')
#model.fit(train_samples[0], train_samples[1], validation_split=0.2, shuffle=True, nb_epoch=7)

history_object=model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=5)

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

model.save('model.h5')  


