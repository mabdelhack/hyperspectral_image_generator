#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCon 2018:
Satellite data is for everyone: insights into modern remote sensing research
with open data and Python

"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"   #CPU mode
from glob import glob
from keras.applications.vgg16 import VGG16 as VGG
from keras.applications.densenet import DenseNet201 as DenseNet
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from hyperspectral_image_generator import hyperspectral_image_generator
import pandas as pd
from keras import backend as K
import tensorflow as tf



# variables
path_to_split_datasets = 'path to split dataset'
batch_size = 128
number_of_epochs = 10
steps_per_epoch = 500

class_indices = {'AnnualCrop': 0, 'Forest': 1, 'HerbaceousVegetation': 2,
                 'Highway': 3, 'Industrial': 4, 'Pasture': 5,
                 'PermanentCrop': 6, 'Residential': 7, 'River': 8,
                 'SeaLake': 9}
num_classes = len(class_indices)

# contruct path
path_to_home = os.path.expanduser("~")
path_to_split_datasets = path_to_split_datasets.replace("~", path_to_home)
path_to_train = os.path.join(path_to_split_datasets, "train")
path_to_validation = os.path.join(path_to_split_datasets, "validation")

# parameters for CNN

base_model = VGG(include_top=False,
                 weights=None,
                 input_shape=(64, 64, 13))

# add a global spatial average pooling layer
top_model = base_model.output
top_model = Flatten()(top_model)
# or just flatten the layers
#    top_model = Flatten()(top_model)
# let's add a fully-connected layer

# only in VGG19 a fully connected nn is added for classfication
# DenseNet tends to overfitting if using additionally dense layers
top_model = Dense(2048, activation='relu')(top_model)
top_model = Dense(1024, activation='relu')(top_model)
# and a logistic layer
predictions = Dense(num_classes, activation='softmax')(top_model)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# print network structure
model.summary()

results = pd.DataFrame(columns=['flip', 'rotation', 'zoom', 'shift', 'shear', 'noising', 'epoch',
                                'training_loss', 'training_accuracy', 'validation_loss', 'validation_accuracy'])

# hyperparameter search cases
training_parameters = list()
training_parameters.append({'flip': False,
                            'zoom': 1.0,
                            'shift': 0.0,
                            'rotation': 0.0,
                            'sheer': 0.0,
                            'noising': None})

# defining ImageDataGenerators
# ... initialization for training
training_files = glob(path_to_train + "/**/*.tif")

# ... initialization for validation
validation_files = glob(path_to_validation + "/**/*.tif")
validation_generator = hyperspectral_image_generator(validation_files, class_indices,
                                                     batch_size=batch_size,
                                                     image_mean='image_mean_std.txt')
Wsave = model.get_weights()

for parameter_set in training_parameters:
    model.set_weights(Wsave)
    train_generator = hyperspectral_image_generator(training_files, class_indices,
                                                    batch_size=batch_size,
                                                    image_mean='image_mean_std.txt',
                                                    rotation_range=parameter_set['rotation'],
                                                    horizontal_flip=parameter_set['flip'],
                                                    vertical_flip=parameter_set['flip'],
                                                    speckle_noise=parameter_set['noising'],
                                                    shear_range=parameter_set['sheer'],
                                                    scale_range=parameter_set['zoom'],
                                                    transform_range=parameter_set['shift']
                                                    )
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # generate callback to save best model w.r.t val_categorical_accuracy
    file_name = "vgg"
    checkpointer = ModelCheckpoint("../data/models/" + file_name +
                                   "_ms_from_scratch." +
                                   "flip{flip}_rot{rotation}_zoom{zoom}_shift{shift}_shear{shear}_noise{noise}".format(
                                       flip='on' if parameter_set['flip'] else 'off',
                                       rotation=parameter_set['rotation'],
                                       zoom=parameter_set['zoom'],
                                       shift=parameter_set['shift'],
                                       shear=parameter_set['sheer'],
                                       noise=parameter_set['noising'] if parameter_set['noising'] is not None else '0') +
                                   "_{epoch:02d}-{val_categorical_accuracy:.3f}." +
                                   "hdf5",
                                   monitor='val_categorical_accuracy',
                                   verbose=1,
                                   save_best_only=False)

    model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=number_of_epochs,
            callbacks=[checkpointer],
            validation_data=validation_generator,
            validation_steps=5)
    history = model.history
    for idx in range(len(history.epoch)):
        epoch = history.epoch[idx]
        training_loss = history.history['loss'][idx]
        validation_loss = history.history['val_loss'][idx]
        training_accuracy = history.history['categorical_accuracy'][idx]
        validation_accuracy = history.history['val_categorical_accuracy'][idx]
        results = results.append({'flip': 1 if parameter_set['flip'] else 0,
                                  'rotation': parameter_set['rotation'],
                                  'zoom': parameter_set['zoom'],
                                  'shift': parameter_set['shift'],
                                  'shear': parameter_set['sheer'],
                                  'noising': parameter_set['noising'] if parameter_set['noising'] is not None else '0',
                                  'epoch': epoch,
                                  'training_loss': training_loss,
                                  'training_accuracy': training_accuracy,
                                  'validation_loss': validation_loss,
                                  'validation_accuracy': validation_accuracy
                                  },
                                 ignore_index=True)
    results.to_csv('results.csv', index=False)
