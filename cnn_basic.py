#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Eoin Devlin
"""

from keras import layers
from keras import models
from keras import losses
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

Train_DIR = '/Users/eoindevlin/Google Drive/3U Masters/DCU/Computer Vision (EE544)/codebook/food101_selected/train'
Test_DIR = '/Users/eoindevlin/Google Drive/3U Masters/DCU/Computer Vision (EE544)/codebook/food101_selected/test'
Validation_DIR = '/Users/eoindevlin/Google Drive/3U Masters/DCU/Computer Vision (EE544)/codebook/food101_selected/validation'
Random_DIR = '/Users/eoindevlin/Google Drive/3U Masters/DCU/Computer Vision (EE544)/codebook/food101_selected/random'

Image_Size = 200 # Size of input images to be scaled to
num_train_images = 6000 
batch_size = 100
num_epochs = 30
steps_per_epoch = num_train_images/batch_size

# Instantiation of a small CNN 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (Image_Size, Image_Size, 3))) # The number of output filters in the convolution is 32. Kernel size = 3x3. 
model.add(layers.MaxPooling2D((2,2))) # Factors by which to downscale in both the vertical and horizontal direction
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten()) # Flattening prior to the initial dot product with kernel
model.add(layers.Dropout(0.5)) # Dropping a neuron with 0.5 probability gets the highest variance for this distribution
model.add(layers.Dense(512, activation='relu')) # Dimensionality of the output space is 512
model.add(layers.Dense(6, activation='sigmoid')) # 6 classes 

model.summary()

# Configuring the model for trainig for categorical classification
model.compile(
        loss=losses.categorical_crossentropy, 
        optimizer=optimizers.RMSprop(lr=1e-4), 
        metrics=['acc'])

# Rescaling images by 1./255 using ImageDataGenerator
train_generator = ImageDataGenerator(rescale=1./255, 
                                     rotation_range=40, 
                                     width_shift_range=0.2, 
                                     height_shift_range=0.2, 
                                     shear_range=0.2, 
                                     zoom_range=0.2, 
                                     horizontal_flip=True).flow_from_directory(Train_DIR, 
                                                         target_size=(Image_Size,Image_Size), 
                                                         batch_size=batch_size, 
                                                         class_mode='categorical')

validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        Validation_DIR, 
        target_size=(Image_Size,Image_Size), 
        batch_size=batch_size, 
        class_mode='categorical')

# Trains the model for a fixed number of epochs
hist = model.fit_generator(
        train_generator, 
        steps_per_epoch=steps_per_epoch, 
        epochs=num_epochs,
        validation_data=validation_generator, 
        validation_steps=50)

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epochs)

# Plot training loss vs. validation loss and training accuracy vs. validation accuracy
fig1=plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Vs. Validation Loss')
plt.grid(True)
plt.legend(['Training', 'Validation'])
plt.style.use(['classic'])
fig1.savefig('loss.png')

fig2=plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Vs. Validation Accuracy')
plt.grid(True)
plt.legend(['Training', 'Validation'], loc='upper left')
plt.style.use(['classic'])
fig2.savefig('acc.png')

model.save('food_small_nonbasic.h5') # Save model

# Tests the model and returns the loss value & metric values for the model in test mode
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        Test_DIR, 
        target_size=(Image_Size,Image_Size), 
        batch_size=batch_size, 
        class_mode='categorical')

# Evaluates model on the test data
results = model.evaluate_generator(
        test_generator, 
        steps=1000)

print('Final Test Accuracy: ', (results[1]*100.00))