# Seefood: Binary Image Classifier for Hot Dogs 

## Overview

This is a classifier which predicts whether a given image is a hot dog. The weights for the model is saved in the .hdf5 file.
The classifier was built using transfer learning on the Inception V3 model and fine tuned with additional images. 

## Usage

Script to scrape images is not included, but should not be hard to find. We scraped around 2000 hot dog images and sampled around 4600 not hot dog images from ImageNet. 
When sampling the "not X" class, make sure to get a good variety of images so that the classifier does not make decisions solely based on attributes such as color or size. 
A sample of images from both groups can be found from the data folder. 

1. Use the get_data.py to create the training set from the scraped images.
2. Use the fit_model.py to leverage Inception V3 for transfer learning. 

The output layer from Inception V3 will be replaced by a layer relevant to hot dogs. 

Understandably, we took a excellent model and turned it into an useless one. 

Ideation is from the show Silicon Valley. 
 
