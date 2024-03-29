# Hyperspectral Image Generator
[![DOI](https://zenodo.org/badge/254969298.svg)](https://zenodo.org/badge/latestdoi/254969298)

This is a tool that creates an image generator for Keras that is useful for hyperspectral images. It implements data augmentation. This tool is developed originally from [CNN Sentinel](https://github.com/jensleitloff/CNN-Sentinel) repository and uses main functions as is from the original repository.
Example implementation in this [paper](http://arxiv.org/abs/2003.13502).

## Features
It allows an image generator to make the following transformations on images:
- Horizontal/Vertical flip
- Rotation
- Shear
- Translation
- Zooming
- Addition of speckle noise

UPDATE: It now supports cropping from larger tiles.

It uses the edge pixels to pad the pixels that become missing due to the transformations.

**Without augmentation:**

![](images_for_notebook/no_augmentation.png)

**With augmentation:**

![](images_for_notebook/augmentation.png)


## Requirements
These are the requirements for running this whole simulation. The generator itself only uses scikit-image and numpy libraries.
- Keras
- Tensorflow
- Numpy
- Pandas
- Scikit-image

For tile cropping:
- Geopandas
- Rasterio
- Fiona
- Shapely

## How to use (basic mode)
The tool itself is located in hyperspectral_image_generator.py. It depends on some preprocessing functions that loads image mean and std and the files.

A use case is imported from the [original repository](https://github.com/jensleitloff/CNN-Sentinel) for CNN-Sentinel classification. You should refer to the documentation there for more details on how to obtain the data.

First, download the [data](http://madm.dfki.de/downloads) and write the data path into split_data_to_train_and_validation.py then run it to generate training and test datasets.

Then run train_hyperspectral_vgg19.py to train a VGG19 network using image augmentation. 

You can test the function to visualize the output using augmentation_test.py.

## How to use tile cropping
![](images_for_notebook/jp2_generator.png)
I used data from the park location in the city of St. Louis, Missouri, USA released by the 
[St. Louis government website](https://www.stlouis-mo.gov/) with [Google maps](maps.google.com) satellite imagery for background.
A demo of the generator is included in slicing_test.py. Note that you need to download the [data](https://doi.org/10.6084/m9.figshare.21431082) and include them in the folder ```images_for_notebook``` (Data is quite large). The shape file with labels 
Centroids are included in the repo (shape files) but it is just a processed version of the original file, I do not possess any rights to the data it comprises.

## Future plans
- Allow different modes of noise other than speckle noise.
