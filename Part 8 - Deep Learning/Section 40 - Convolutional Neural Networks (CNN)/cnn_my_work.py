# -*- coding: utf-8 -*-
"""
Created on Thu May 24 23:52:36 2018

@author: Harshit Maheshwari
"""

# Convolutional Neural Network

# Importing the Keras libraries
from keras.models import Sequential
from keras.layers import Conv2D #For images since images are in 2D, videos in 3D with time the third dimension
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3))) # 32 feature dectors with 3*3 (32, 3, 3) matrix size
              
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding another layer
classifier.add(Conv2D(32, (3, 3), activation='relu')) # Another layer added to imporve accuracy
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', 
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
       
classifier.fit_generator(training_set,
                         steps_per_epoch=8000, #Number of images in training set
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000) #Numbe of images in test set