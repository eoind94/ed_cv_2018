#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eoin Devlin
"""

from keras import layers
from keras import models
from keras import losses
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

import matplotlib.pyplot as plt
import numpy as np

Train_DIR = '/Users/eoindevlin/Google Drive/3U Masters/DCU/Computer Vision (EE544)/codebook/food101_selected/train'
Test_DIR = '/Users/eoindevlin/Google Drive/3U Masters/DCU/Computer Vision (EE544)/codebook/food101_selected/test'
Validation_DIR = '/Users/eoindevlin/Google Drive/3U Masters/DCU/Computer Vision (EE544)/codebook/food101_selected/validation'

Image_Size = 200 # Size of input images to be scaled to 

num_epochs = 30
batch_size = 20

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(Image_Size, Image_Size, 3))

conv_base.summary()

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 6, 6, 512))
    labels = np.zeros(shape=(sample_count, 6))
    generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory, target_size=(Image_Size, Image_Size), batch_size = batch_size, class_mode='categorical')
    

    i = 0

    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * 20 : (i + 1) * 20] = features_batch
        labels[i * 20 : (i + 1) * 20] = labels_batch
        i += 1
        if (i * 20) >= sample_count:
            break
        
    return features, labels

train_features, train_labels = extract_features(Train_DIR, 6000)
validation_features, validation_labels = extract_features(Validation_DIR, 3000)
test_features, test_labels = extract_features(Test_DIR, 3000)

train_features = np.reshape(train_features, (6000, 6 * 6 * 512))
validation_features = np.reshape(validation_features, (3000, 6 * 6 * 512))
test_features = np.reshape(test_features, (3000, 6 * 6 * 512))


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=6 * 6 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss=losses.categorical_crossentropy,
              metrics=['acc'])
    
hist = model.fit(train_features, train_labels,
                 epochs = num_epochs,
                 batch_size = batch_size,
                 verbose = 2,
                 validation_data = (validation_features, validation_labels))

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

model.save('food_pretrained.h5') # Save model